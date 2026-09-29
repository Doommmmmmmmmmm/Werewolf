from __future__ import annotations

import asyncio, hashlib, inspect, json, re
from collections.abc import Mapping
from ..core.errors import ModelClientError, RuleViolationError
from .llm.coordinator import ModelRequestCoordinator
from ..prompts import RoleProfile, render_prompt

CN=re.compile(r"[\u3400-\u9fff]"); PID_NUM=re.compile(r"(\d+)$"); PID_WORD=re.compile(r"\b[a-zA-Z_]*\d+\b")
CURRENT_ROUND_DIALOGUE_TOOL_NAME="read_current_round_dialogue"
DEFAULT_MAX_TOOL_CALLS_PER_DECISION=5; DEFAULT_MAX_TOOL_RESULT_TOKENS=1000; DEFAULT_MAX_PROMPT_CHARS=12000; DEFAULT_MAX_DECISION_RETRIES=2
SPEECH=frozenset({"speak","last_words"}); NLIM=84; LLIM=8; SLIM=1600
CURRENT_ROUND_DIALOGUE_TOOL={"name":CURRENT_ROUND_DIALOGUE_TOOL_NAME,"description":"读取当前轮依法可见发言。","parameters":{"type":"object","properties":{},"additionalProperties":False}}


def _acts(req):
    xs=req.get("allowed_actions"); return [x for x in xs if isinstance(x,Mapping)] if isinstance(xs,list) else []

def _targets(req):
    r=[]
    for a in _acts(req):
        ts=a.get("target_ids")
        if isinstance(ts,list):
            for t in ts:
                if isinstance(t,(str,int)) and not isinstance(t,bool) and str(t) not in r: r.append(str(t))
    return r

def _phase(req):
    return (f"{req.get('phase','')} {req.get('channel','')} {req.get('kind','')} "+",".join(str(a.get("kind") or "") for a in _acts(req))).lower()


def render_villager_decision_checklist(request):
    p=_phase(request); xs=["【平民清单】区分系统事实/玩家声明/推测；优先查验链、警徽链、死亡、票型。","不编造系统事实；无绝对收益不要空跳猎人/女巫/守卫等强神。"]
    if "vote" in p or "投票" in p: xs += ["投票只选allowed_actions.target_ids；站边明确预言家、跟可信查杀、保护金水，少弃票。","至少比较两人：查杀/事实错/票型异常/发言矛盾者优先出。"]
    if any(x in p for x in ("speak","discussion","发言","day","last_words")): xs.append("发言给出站边、关注对象、暂不投对象；闭眼视角表达，不乱带强神身份。")
    if any(x in p for x in ("sheriff","警长","警徽")): xs.append("警长投票只投合法候选；不推荐退水/不上警者，优先报验清晰且警徽流可靠者。")
    return "\n".join(xs)


def render_action_contract(request):
    xs=["【最终JSON】只输出一个JSON对象，不加解释/Markdown/额外字段。",f"阶段:{request.get('phase') or 'unknown'}；频道:{request.get('channel') or 'unknown'}。"]
    if not _acts(request): return "\n".join(xs+["没有allowed_actions，不能臆造行动。"])
    for a in _acts(request):
        k=str(a.get("kind") or "")
        if not k: continue
        ts=[str(x) for x in a.get("target_ids",[]) if isinstance(x,(str,int)) and not isinstance(x,bool)]
        ex={"kind":k}
        if k in SPEECH: ex["text"]="我按公开查验和票型站边，先关注矛盾位。"
        elif ts: ex["target_id"]=ts[0]
        xs.append("- "+json.dumps(ex,ensure_ascii=False,separators=(",",":")))
        if ts: xs.append("  target_id只能是:"+"、".join(ts))
        if k in SPEECH:
            cs=[]
            if a.get("max_chars") is not None: cs.append(f"text≤{a['max_chars']}字")
            if a.get("require_chinese"): cs.append("至少含中文")
            if cs: xs.append("  "+"；".join(cs))
    return "\n".join(xs+["kind/target_id必须用英文协议值。"])


def normalize_decision(raw, request):
    s=raw.get("decision",raw) if isinstance(raw,dict) else {}; a={"request_id":request["request_id"],"player_id":request["player_id"],"kind":str(s.get("kind",s.get("action","")))}
    t=s.get("target_id",s.get("targetId"))
    if t is not None and t!="": a["target_id"]=str(t)
    if isinstance(s.get("text"),str): a["text"]=s["text"]
    return a


def decision_error(action, request):
    ok=next((a for a in request["allowed_actions"] if a["kind"]==action["kind"]),None)
    if ok is None: return "kind 必须是 allowed_actions 中的一项"
    if action["kind"] in SPEECH:
        text=str(action.get("text","")).strip()
        if not text: return f"{action['kind']} 必须提供非空 text"
        if len(text)>int(ok["max_chars"]): return f"发言超过 max_chars={ok['max_chars']}"
        if ok.get("require_chinese") and not CN.search(text): return "发言必须包含中文"
    ts=ok.get("target_ids")
    if ts and action.get("target_id") not in {str(x) for x in ts}: return "target_id 必须是 target_ids 中的一项"
    if not ts and action.get("target_id"): return "该行动不能提供 target_id"
    return None


class TaskAgent:
    def __init__(self, *, player_id:str, profile:RoleProfile, model_client, persona="", request_coordinator:ModelRequestCoordinator|None=None, max_tokens=900, max_decision_retries=DEFAULT_MAX_DECISION_RETRIES, max_tool_calls_per_decision=DEFAULT_MAX_TOOL_CALLS_PER_DECISION, max_tool_result_tokens=DEFAULT_MAX_TOOL_RESULT_TOKENS, max_prompt_chars=DEFAULT_MAX_PROMPT_CHARS):
        if not hasattr(model_client,"complete_json"): raise ValueError("TaskAgent 需要具有 complete_json 的模型客户端")
        self.player_id=str(player_id); self.profile=profile; self.model_client=model_client; self.persona=persona; self.request_coordinator=request_coordinator
        self.max_tokens=min(900,int(max_tokens)); self.max_decision_retries=max(0,int(max_decision_retries)); self.max_tool_calls_per_decision=min(5,max(0,int(max_tool_calls_per_decision)))
        self.max_tool_result_tokens=min(1000,max(1,int(max_tool_result_tokens))); self.max_prompt_chars=min(8000,max(1,int(max_prompt_chars)))
        self._model_token_usage={k:0 for k in ("successful_response_count","api_attempt_count","reported_usage_response_count","input_tokens","output_tokens","total_tokens")}
        self._public_ledger_game_id=None; self._reset_public_ledger(None)

    async def observe(self, sync_packet):
        if isinstance(sync_packet,Mapping):
            gid=self._game_id(sync_packet)
            if gid is not None and gid!=self._public_ledger_game_id: self._reset_public_ledger(gid)
            self._ingest(sync_packet)

    async def decide(self, turn_packet):
        if turn_packet["self"]["player_id"]!=self.player_id: raise ValueError("行动包被发送给了错误玩家")
        private=turn_packet["private_information"]
        if str(private.get("role"))!=self.profile.role: raise ValueError("Task-Agent 的角色档案与行动包身份不一致")
        gid=self._game_id(turn_packet)
        if gid is not None and gid!=self._public_ledger_game_id: self._reset_public_ledger(gid)
        self._ingest(turn_packet.get("public_state")); req=turn_packet["request"]; dialogue=(turn_packet.get("tool_context") or {}).get("current_round_dialogue") or []
        def tool(name, arguments):
            del arguments
            if name!=CURRENT_ROUND_DIALOGUE_TOOL_NAME: return {"error":f"不支持的工具：{name}"}
            g=turn_packet.get("game") or {}; return {"round":g.get("round"),"phase":g.get("public_phase",g.get("phase")),"dialogue":dialogue}
        feedback=""
        for i in range(self.max_decision_retries+1):
            use_tool=i==0 and self.max_tool_calls_per_decision>0; msg={"instruction":self._turn_instruction(turn_packet,feedback),"packet":self._packet(turn_packet)}
            try:
                raw=await self._complete_json(system=self._system_prompt(private),messages=[{"role":"user","content":json.dumps(msg,ensure_ascii=False)}],tools=[CURRENT_ROUND_DIALOGUE_TOOL] if use_tool else None,tool_executor=tool if use_tool else None,max_tool_calls=self.max_tool_calls_per_decision if use_tool else 0,max_tool_result_tokens=self.max_tool_result_tokens,max_prompt_chars=self.max_prompt_chars)
            except ModelClientError:
                if i>=self.max_decision_retries: raise
                feedback="上一轮模型调用未产生可用JSON，请重答。"; continue
            self._record_token_usage(raw); action=normalize_decision(raw,req); err=decision_error(action,req)
            if err is None: return action
            if i>=self.max_decision_retries: raise RuleViolationError(f"模型返回非法行动：{err}",attempted_action=action,validation_error=err)
            feedback=err
        raise RuleViolationError("模型未能返回合法行动",attempted_action={},validation_error="unknown")

    def agent_manifest(self):
        return {"agent_type":"task_agent","player_id":self.player_id,"role":self.profile.role,"conversation_mode":"new_session_per_decision","tool_name":CURRENT_ROUND_DIALOGUE_TOOL_NAME,"max_tool_calls_per_decision":self.max_tool_calls_per_decision,"max_tool_result_tokens":self.max_tool_result_tokens,"max_prompt_chars":self.max_prompt_chars,"max_decision_retries":self.max_decision_retries,"task_fingerprint":hashlib.sha256(self.profile.task.encode()).hexdigest()}
    def model_token_usage_snapshot(self): return dict(self._model_token_usage)
    def _system_prompt(self, private):
        return render_prompt("player_system.txt",player_id=self.player_id,role=private["role"],team=private["team"],persona=self.persona,role_base=self.profile.base,role_task=self.profile.task,tool_name=CURRENT_ROUND_DIALOGUE_TOOL_NAME,max_tool_calls=self.max_tool_calls_per_decision,max_tool_result_tokens=self.max_tool_result_tokens,max_prompt_chars=self.max_prompt_chars)
    def _turn_instruction(self,tp,fb):
        req=tp["request"]; extra=f"\n上次非法:{fb}；只按契约输出。" if fb else ""
        return render_prompt("player_turn_instruction.txt",action_contract=render_action_contract(req),validation_feedback=self._summary(tp,req)+"\n"+render_villager_decision_checklist(req)+extra)
    async def _complete_json(self,**kw):
        if self.request_coordinator is not None: return await self.request_coordinator.complete_json(self.model_client,max_tokens=self.max_tokens,**kw)
        fn=self.model_client.complete_json; return await fn(max_tokens=self.max_tokens,**kw) if inspect.iscoroutinefunction(fn) else await asyncio.to_thread(fn,max_tokens=self.max_tokens,**kw)
    def _record_token_usage(self,r):
        self._model_token_usage["successful_response_count"]+=1; self._model_token_usage["api_attempt_count"]+=max(1,self._nonneg(getattr(r,"api_attempts",1),1) or 1); u=getattr(r,"token_usage",None)
        if isinstance(u,dict):
            vals={k:self._nonneg(u.get(k),None) for k in ("input_tokens","output_tokens","total_tokens")}
            if any(v is not None for v in vals.values()):
                self._model_token_usage["reported_usage_response_count"]+=1
                for k,v in vals.items():
                    if v is not None: self._model_token_usage[k]+=v
    @staticmethod
    def _nonneg(v, fallback):
        if isinstance(v,bool): return fallback
        try: n=int(v)
        except (TypeError,ValueError): return fallback
        return n if n>=0 else fallback

    def _reset_public_ledger(self, game_id):
        self._public_ledger_game_id=game_id; self._public_ledger={"alive":set(),"dead":set(),"known":set(),"claims":[],"votes":[],"sheriff":[],"deaths":[],"withdrawn":set()}
    def _packet(self,tp):
        g=tp.get("game") if isinstance(tp.get("game"),Mapping) else {}
        return {"game":{k:g.get(k) for k in ("round","phase","public_phase") if k in g},"self":tp.get("self"),"private_information":tp.get("private_information"),"request":tp.get("request"),"public_ledger":self._snapshot(tp.get("public_state")),"history_policy":{"default_context":"compact_public_ledger_only","current_round_dialogue_tool":CURRENT_ROUND_DIALOGUE_TOOL_NAME,"max_tool_calls":self.max_tool_calls_per_decision,"max_tool_result_tokens":self.max_tool_result_tokens}}
    def _ingest(self,payload):
        if payload is None: return
        notes=self._notes(payload); self._public_ledger["known"].update(self._ids(payload,True))
        if isinstance(payload,Mapping):
            low={str(k).lower():v for k,v in payload.items()}; self._public_ledger["alive"].update(self._collect(low,self.ALIVE)); self._public_ledger["dead"].update(self._collect(low,self.DEAD))
        for n in notes:
            lo=n.lower()
            if "vote" in lo or "票" in n: self._add("votes",n)
            if any(x in lo for x in ("sheriff","badge")) or any(x in n for x in ("警长","警徽")): self._add("sheriff",n)
            if any(x in lo for x in ("death","dead","kill","out")) or any(x in n for x in ("死亡","出局","淘汰","刀")): self._add("deaths",n); self._public_ledger["dead"].update(self._ids_text(n))
            if any(x in n for x in ("退水","不上警","不竞选","弃选")): self._public_ledger["withdrawn"].update(self._ids_text(n))
            if any(x in n for x in ("预言家","金水","查杀","查验","警徽流","自称","claim")): self._add("claims",n)
    def _snapshot(self,ps):
        notes=self._notes(ps); cur=self._ids(ps,True); dead=set(self._public_ledger["dead"])|self._dead_from(notes); alive=(set(self._public_ledger["alive"])|cur)-dead; known=(set(self._public_ledger["known"])|cur|alive|dead)-{self.player_id}
        return {"alive_ids":self._sort(alive),"dead_ids":self._sort(dead),"known_ids":self._sort(known),"claims":self._rel(notes,"claims"),"votes":self._rel(notes,"votes"),"sheriff":self._rel(notes,"sheriff"),"deaths":self._rel(notes,"deaths")}
    def _summary(self,tp,req):
        s=self._snapshot(tp.get("public_state")); notes=list(dict.fromkeys(s["claims"]+s["votes"]+s["sheriff"]+s["deaths"]))[:12]; cand=self._candidate_pool(req,s["alive_ids"],s["known_ids"])
        xs=["【公共台账】","存活:"+self._join(s["alive_ids"]),"已死:"+self._join(s["dead_ids"])]
        if s["claims"]: xs.append("查验/警徽链:"+" | ".join(s["claims"][:3]))
        if s["deaths"]: xs.append("死亡:"+" | ".join(s["deaths"][:2]))
        xs.append(self._vote_summary(cand,s["alive_ids"],s["dead_ids"],notes) if self._is_vote(req) else self._speak_summary(cand,s["alive_ids"],s["dead_ids"],notes))
        return self._trim("\n".join(xs),SLIM)
    def _rel(self,notes,key):
        items=list(self._public_ledger[key])+notes; fs={"claims":("预言家","金水","查杀","查验","警徽流","验","claim"),"votes":("vote","票"),"sheriff":("sheriff","badge","警长","警徽"),"deaths":("death","dead","kill","out","死亡","出局","淘汰")}
        return self._dedupe([self._trim(x,NLIM) for x in items if any(t in x.lower() or t in x for t in fs[key])])[-LLIM:]
    def _candidate_pool(self,req,alive,known):
        allowed=_targets(req); pool=set(allowed or alive or known); pool.discard(self.player_id); pool.difference_update(self._public_ledger["dead"])
        if allowed and self._is_sheriff_vote(req):
            pool.difference_update(self._public_ledger["withdrawn"])
            if not pool: pool=set(allowed)-{self.player_id}
        return self._sort(pool)
    def _vote_summary(self,cand,alive,dead,notes):
        r=self._rank(cand,notes,dead)
        if len(r)<2: r+=self._rank([p for p in alive if p not in {x[0] for x in r}],notes,dead)
        rows=[f"{i}.{p} 分={sc} {why}" for i,(p,sc,why) in enumerate(r[:2],1)] or ["候选不足，按唯一合法目标或公开查验/票型决定。"]
        return "【投票参考】只投合法目标；优先比较:\n"+"\n".join(rows)
    def _speak_summary(self,cand,alive,dead,notes):
        r=self._rank(cand or alive,notes,dead)
        if not r: return "【发言参考】关注:灰区 公开信息不足；不要空跳强神。"
        safe=sorted(r,key=lambda x:(x[1],x[0]))[0]; return f"【发言参考】关注:{r[0][0]} {r[0][2]}；暂不投/保护:{safe[0]} {safe[2]}；不要空跳强神。"
    def _rank(self,cand,notes,dead): return sorted([(p,*self._score(p,notes)) for p in self._sort(cand) if p!=self.player_id and p not in dead],key=lambda x:(-x[1],x[0]))
    def _score(self,p,notes):
        bad=("查杀","悍跳","对跳","爆狼","矛盾","票型异常","冲票","假","狼"); good=("金水","银水","可信","真预","预言家","警徽流","强神","守卫","女巫","猎人"); sc=0; why=[]
        for n in notes:
            if p.lower() not in n.lower(): continue
            if any(x in n for x in bad): sc+=3; why.append("疑点:"+self._trim(n,56))
            elif any(x in n for x in good): sc-=3; why.append("保护:"+self._trim(n,56))
            elif "vote" in n.lower() or "票" in n: sc+=1; why.append("票型:"+self._trim(n,56))
            else: sc+=1; why.append(self._trim(n,56))
        return sc,"；".join(why[:2]) if why else "公开信息不足"
    def _notes(self,payload): out=[]; self._walk(payload,out,0); return self._dedupe(out)[:16]
    def _walk(self,obj,out,depth):
        if obj is None or depth>2 or len(out)>=16: return
        if isinstance(obj,Mapping):
            s=self._snap(obj)
            if s: out.append(s)
            for k,v in list(obj.items())[:12]:
                if str(k).lower() not in {"private","private_information","hidden","secret"}: self._walk(v,out,depth+1)
        elif isinstance(obj,list):
            for v in obj[-12:]: self._walk(v,out,depth+1)
    def _snap(self,m):
        d={str(k).lower():v for k,v in m.items()}; event=self._first(d,("event","event_type","type","kind","action","phase")); player=self._first(d,("player_id","speaker_id","voter_id","actor_id","claimer_id","source_id")); target=self._first(d,("target_id","target","vote_target","to_id","dead_id","dead_player_id")); holder=self._first(d,("sheriff_id","badge_holder","badge_holder_id")); role=self._first(d,("role","claim_role","claimed_role")); text=self._first(d,("text","content","message","reason","speech","claim"))
        if not (text or any(k in d for k in ("voter_id","vote_target","target_id","dead_id","sheriff_id","badge_holder")) or any(t in event.lower() for t in ("vote","sheriff","badge","death","claim","speak"))): return None
        if text and not (player or target or holder or role or any(x in text for x in ("预言家","金水","查杀","查验","警徽","死亡","出局","退水","票","狼"))): return None
        return self._trim(" ".join(x for x in (event,f"player={player}" if player else "",f"target={target}" if target else "",f"sheriff={holder}" if holder else "",f"role={role}" if role else "","text="+text if text else "") if x),NLIM)
    def _ids(self,payload,inc):
        ids=set()
        def walk(o,depth):
            if depth>2: return
            if isinstance(o,Mapping):
                for k,v in list(o.items())[:16]:
                    low=str(k).lower()
                    if low in self.PLAYER_KEYS and self._is_id(v): ids.add(str(v))
                    elif inc and low in self.COLL_KEYS: ids.update(self._id_set(v))
                    else: walk(v,depth+1)
            elif isinstance(o,list):
                for v in o[:16]: walk(v,depth+1)
        walk(payload,0); ids.discard(self.player_id); return ids
    def _collect(self,m,keys):
        ids=set()
        for k,v in m.items():
            if str(k).lower() in keys: ids.update(self._id_set(v))
        return ids
    def _dead_from(self,notes):
        ids=set()
        for n in notes:
            if any(t in n.lower() for t in ("death","dead","kill","out")) or any(t in n for t in ("死亡","出局","淘汰")): ids.update(self._ids_text(n))
        return ids
    @staticmethod
    def _ids_text(text): return {x for x in PID_WORD.findall(text) if len(x)<=16}
    def _id_set(self,value): return {str(x) for x in value if self._is_id(x)} if isinstance(value,(list,tuple,set)) else ({str(value)} if self._is_id(value) else set())
    @staticmethod
    def _is_id(v): return not isinstance(v,bool) and (isinstance(v,int) or (isinstance(v,str) and len(v.strip())<=16 and any(c.isdigit() for c in v)))
    def _add(self,key,value):
        arr=self._public_ledger[key]; value=self._trim(value,NLIM)
        if value and value not in arr: arr.append(value); del arr[:max(0,len(arr)-LLIM)]
    @staticmethod
    def _first(m,keys):
        for k in keys:
            v=m.get(k)
            if isinstance(v,(str,int,float)) and not isinstance(v,bool): return TaskAgent._trim(v,NLIM)
        return ""
    @staticmethod
    def _is_vote(req): p=_phase(req); return "vote" in p or "投票" in p
    @staticmethod
    def _is_sheriff_vote(req): p=_phase(req); return ("vote" in p or "投票" in p) and ("sheriff" in p or "警" in p)
    @staticmethod
    def _trim(text,limit): s=" ".join(str(text).split()); return s if len(s)<=limit else s[:max(0,limit-1)]+"…"
    @staticmethod
    def _dedupe(items): return list(dict.fromkeys(x for x in items if x))
    @staticmethod
    def _join(ids): return "未知" if not ids else "、".join(ids[:12])+(f" 等{len(ids)-12}人" if len(ids)>12 else "")
    @staticmethod
    def _sort(ids):
        def key(v):
            m=PID_NUM.search(v); return (PID_NUM.sub("",v),int(m.group(1)) if m else 10**9,v)
        return sorted(list(dict.fromkeys(str(x) for x in ids if str(x))),key=key)
    @staticmethod
    def _game_id(payload):
        for src in (payload,payload.get("game") if isinstance(payload.get("game"),Mapping) else {}):
            for k in ("game_id","gameid","match_id","matchid","session_id"):
                v=src.get(k)
                if isinstance(v,(str,int)) and not isinstance(v,bool): return str(v)
        return None
    PLAYER_KEYS=frozenset({"player_id","speaker_id","voter_id","claimer_id","actor_id","source_id","target_id","from_id","to_id","badge_holder","badge_holder_id","sheriff_id","dead_id","dead_player_id"})
    COLL_KEYS=frozenset({"alive","alive_ids","alive_player_ids","dead","dead_ids","dead_player_ids","players","player_ids","target_ids","candidates","survivors","living","graveyard"})
    ALIVE=frozenset({"alive","alive_ids","alive_player_ids","survivors","living","live_ids","current_alive"})
    DEAD=frozenset({"dead","dead_ids","dead_player_ids","eliminated","casualties","deaths","graveyard"})
