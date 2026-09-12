"""Optional ordinary-Python scheduler: wait for next local HH:MM, then exec runner.

Run explicitly under nohup. Merely installing this file does not schedule a job.
"""
import argparse
from datetime import datetime,timedelta
import os
from pathlib import Path
import time


def next_start(clock,now=None):
    now=now or datetime.now();hour,minute=map(int,clock.split(':'))
    target=now.replace(hour=hour,minute=minute,second=0,microsecond=0)
    return target if target>now else target+timedelta(days=1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--time',default='02:30');p.add_argument('--config',required=True)
    p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=1)
    args=p.parse_args();target=next_start(args.time)
    print(f'Waiting until {target.astimezone().isoformat()} (host local timezone); no numerical work has started',flush=True)
    while datetime.now()<target:time.sleep(min(30,max(.1,(target-datetime.now()).total_seconds())))
    repo=Path(__file__).resolve().parent.parent;os.chdir(repo)
    os.execv('/bin/bash',['/bin/bash',str(repo/'scripts/run_historical.sh'),args.config,args.output,str(args.workers)])


if __name__=='__main__':main()
