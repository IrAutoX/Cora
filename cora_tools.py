import subprocess,sys,platform,datetime
TOOLS={"system_info":lambda args:f"{platform.system()} {platform.release()} | Python {platform.python_version()}","time":lambda args:datetime.datetime.now().astimezone().isoformat()}
def python_tool(args):
    if not args:return "usage: python <code>"
    r=subprocess.run([sys.executable,"-c"," ".join(args)],capture_output=True,text=True,timeout=5)
    return (r.stdout+r.stderr).strip()[:4000]
TOOLS["python"]=python_tool
def list_tools(): return sorted(TOOLS)
def run_tool(name,args):
    if name not in TOOLS:return "tool not enabled: "+name
    try:return TOOLS[name](args)
    except Exception as e:return "tool error: "+str(e)
