from __future__ import annotations
import asyncio,json,re,sys,time,urllib.request,urllib.error
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parent
TOOLS_DIR=ROOT/"Runtime"/"Tools"/"dexter-tools"
URL="http://127.0.0.1:11434/api/chat"
MODEL="gemma4-jarvis:12b"
SEED=424242
PROMPTS=[
"What's fifteen percent of 200?",
"Why does Windows use so much RAM even when I'm not doing much?",
"you alive?",
"Explain why SSDs slow down when they're nearly full.",
]
SYSTEM="""You are Dexter, Ahmed's local computer assistant.

Speak naturally, familiarly, and with a dry, understated personality.

Default to a short reply: usually 1-3 sentences. Expand only when Ahmed asks for detail or the question genuinely cannot be answered well that briefly.

Use ordinary conversational prose by default. Acknowledge completed actions naturally rather than like status telemetry. Be direct when Ahmed is mistaken. Avoid generic assistant pleasantries."""

def dump(x:Any)->dict:
    if hasattr(x,"model_dump"): return x.model_dump(by_alias=True,exclude_none=True)
    if hasattr(x,"dict"): return x.dict(by_alias=True,exclude_none=True)
    raise TypeError(type(x))

async def get_tools():
    if not TOOLS_DIR.is_dir():
        raise RuntimeError(f"Missing {TOOLS_DIR}; put this script in Jarvis repo root.")
    sys.path.insert(0,str(TOOLS_DIR))
    try: import server
    except Exception as e:
        raise RuntimeError("Import failed. Run with dexter-tools venv.\n"+repr(e))
    f=getattr(server.mcp,"list_tools",None)
    if f:
        v=f()
        if hasattr(v,"__await__"): v=await v
        return list(v)
    return list(server.mcp._tool_manager.list_tools())

def ollama_tool(t):
    d=dump(t)
    s=d.get("inputSchema") or d.get("input_schema") or {"type":"object","properties":{}}
    return {"type":"function","function":{
        "name":d["name"],"description":str(d.get("description") or ""),"parameters":s}}

def call(prompt,tools):
    payload={"model":MODEL,"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],
             "stream":False,"think":False,"keep_alive":"5m","options":{"seed":SEED}}
    if tools is not None: payload["tools"]=tools
    req=urllib.request.Request(URL,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"},method="POST")
    t=time.perf_counter()
    try:
        with urllib.request.urlopen(req,timeout=300) as r: out=json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP {e.code}: "+e.read().decode(errors="replace"))
    return payload,out,time.perf_counter()-t

def metrics(text):
    ls=[x.strip() for x in text.splitlines() if x.strip()]
    return {
      "words":len(re.findall(r"\S+",text)),
      "heading":any(re.match(r"^#{1,6}\s+",x) for x in ls),
      "numbered":any(re.match(r"^\d+[.)]\s+",x) for x in ls),
      "bullets":any(re.match(r"^[-*+]\s+",x) for x in ls),
      "bold_label":any(re.match(r"^\*\*[^*]+\*\*:?",x) for x in ls),
    }

def main():
    raw=asyncio.run(get_tools()); tools=[ollama_tool(t) for t in raw]
    desc=sum(len(str(dump(t).get("description") or "")) for t in raw)
    sch=sum(len(json.dumps(dump(t).get("inputSchema") or dump(t).get("input_schema") or {},separators=(",",":"))) for t in raw)
    print("="*78); print("DEXTER REAL-TOOLS STYLE ISOLATION PROBE"); print("="*78)
    print(f"Model: {MODEL} | Seed: {SEED} | Goose: ABSENT | Gateway: ABSENT")
    print(f"Live tools: {len(tools)} | description chars: {desc} | schema chars: {sch}")
    results=[]
    for name,ts in [("A_MINIMAL_NO_TOOLS_CONTROL",None),("B_MINIMAL_WITH_REAL_TOOLS",tools)]:
        print("\n"+"#"*78+"\n"+name+"\n"+"#"*78)
        for i,q in enumerate(PROMPTS,1):
            payload,out,elapsed=call(q,ts)
            msg=out.get("message") or {}; text=msg.get("content",""); calls=msg.get("tool_calls") or []; m=metrics(text)
            print("\n"+"-"*78); print(f"[{name} / {i}] USER: {q}\n\nASSISTANT:\n{text or '<no text>'}")
            if calls: print("\nTOOL CALLS:\n"+json.dumps(calls,indent=2,ensure_ascii=False))
            print(f"\nMETRICS: words={m['words']} heading={m['heading']} numbered={m['numbered']} bullets={m['bullets']} bold_label={m['bold_label']} elapsed={elapsed:.2f}s prompt_tokens={out.get('prompt_eval_count')} output_tokens={out.get('eval_count')} tool_calls={len(calls)}")
            results.append({"condition":name,"prompt":q,"request":payload,"response":out,"metrics":m,"seconds":elapsed})
    stamp=datetime.now().strftime("%Y%m%d-%H%M%S")
    Path(f"tools_style_probe_results_{stamp}.json").write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding="utf-8")
    Path(f"tools_style_probe_catalog_{stamp}.json").write_text(json.dumps(tools,indent=2,ensure_ascii=False),encoding="utf-8")
    print("\nSaved JSON results and exact Ollama-formatted tool catalog in repo root.")

if __name__=="__main__":
    try: main()
    except Exception as e:
        print("\nERROR:",e,file=sys.stderr); raise SystemExit(1)
