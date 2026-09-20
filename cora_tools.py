import json,os,subprocess,sys,urllib.parse,urllib.request
from datetime import datetime,timezone
def system_info(_=""): return json.dumps({"python":sys.version.split()[0],"platform":sys.platform,"cwd":os.getcwd()},ensure_ascii=False)
def current_time(_=""): return datetime.now(timezone.utc).astimezone().isoformat()
def web_get(url):
    if not (url.startswith("https://") or url.startswith("http://")): raise ValueError("URL must start with http:// or https://")
    r=urllib.request.Request(url,headers={"User-Agent":"Cora/1.0"})
    with urllib.request.urlopen(r,timeout=12) as x: return x.read(12000).decode("utf-8","replace")
def web_answer(q):
    obj=json.loads(web_get("https://api.duckduckgo.com/?q="+urllib.parse.quote(q)+"&format=json&no_html=1"))
    return json.dumps({"abstract":obj.get("AbstractText",""),"url":obj.get("AbstractURL",""),"related":obj.get("RelatedTopics",[])[:5]},ensure_ascii=False)
def python_exec(code):
    p=subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,timeout=8); return json.dumps({"returncode":p.returncode,"stdout":p.stdout[-12000:],"stderr":p.stderr[-12000:]},ensure_ascii=False)
def install_package(name):
    if not name or any(x in name for x in ";&|<>$"): return "نام پکیج نامعتبر است."
    p=subprocess.run([sys.executable,"-m","pip","install",name],capture_output=True,text=True,timeout=120); return json.dumps({"returncode":p.returncode,"stdout":p.stdout[-6000:],"stderr":p.stderr[-6000:]},ensure_ascii=False)
TOOLS={"system_info":system_info,"time":current_time,"web_get":web_get,"web_answer":web_answer,"python":python_exec,"install":install_package}
def list_tools(): return sorted(TOOLS)
def run_tool(name,arg=""):
    if name not in TOOLS: return "Unknown tool: "+name
    try: return TOOLS[name](arg)
    except Exception as e: return "Tool error: "+type(e).__name__+": "+str(e)
