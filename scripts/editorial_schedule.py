#!/usr/bin/env python3
"""Plan and verify resilient editorial automation without relying on exact cron strings."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
ET=ZoneInfo("America/New_York")


def load(path,default):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def stamp(value):
    if not value:return None
    try:
        text=str(value).replace("Z","+00:00")
        parsed=datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (ValueError,TypeError):return None


def age_hours(value,now):
    parsed=stamp(value)
    if not parsed:return None
    return (now-parsed.astimezone(timezone.utc)).total_seconds()/3600


def same_et_day(value,now):
    parsed=stamp(value)
    return bool(parsed and parsed.astimezone(ET).date()==now.astimezone(ET).date())


def today_state(root,now):
    day=now.astimezone(ET).date().isoformat()
    return load(Path(root)/"docs/editorial/runs"/f"{day}.json",{})


def today_catalog(root,now):
    day=now.astimezone(ET).date().isoformat()
    catalog=load(Path(root)/"docs/editorial/published.json",[])
    return [row for row in catalog if row.get("date")==day and row.get("kind")=="Analysis"]


def config(root):
    return load(Path(root)/"config/editorial.json",{})


def writer_need(root,now):
    cfg=config(root)
    limit=min(int(cfg.get("writer",{}).get("daily_story_limit",2) or 2),2)
    if not cfg.get("writing_enabled",False):
        return False,"writing_disabled"
    published=today_catalog(root,now)
    if len(published)>=limit:
        return False,"daily_publication_limit_reached"
    state=today_state(root,now)
    if state.get("funding_required"):
        return False,"funding_required"
    slots=state.get("slots",{})
    if any(value.get("status")=="started" for value in slots.values()):
        return False,"uncertain_started_slot"
    allocation=state.get("allocation",[])
    if len(allocation)<limit:
        return True,"allocation_incomplete"
    for index,(sport,_) in enumerate(allocation[:limit]):
        status=slots.get(f"{index}-{str(sport).lower()}",{}).get("status")
        if status in (None,"waiting_for_data"):
            return True,f"slot_{index}_{status or 'unattempted'}"
    return False,"daily_slots_terminal"


def mlb_retry_relevant(root,now):
    state=today_state(root,now)
    if not state:
        return True,"first_daily_attempt"
    allocation=state.get("allocation",[])
    slots=state.get("slots",{})
    for index,(sport,_) in enumerate(allocation):
        if sport!="MLB":continue
        status=slots.get(f"{index}-mlb",{}).get("status")
        if status in (None,"waiting_for_data"):
            return True,f"mlb_slot_{status or 'unattempted'}"
    reason=str(state.get("data_skips",{}).get("MLB","")).lower()
    if any(term in reason for term in ("refresh","stale","model inputs","history","current-day")):
        return True,"mlb_data_skip_recoverable"
    return False,"no_mlb_retry_needed"


def plan(root=ROOT,now=None,event_name=None,event_schedule=None,manual_refresh=False):
    now=(now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    local=now.astimezone(ET)
    event_name=event_name or os.getenv("GITHUB_EVENT_NAME","")
    event_schedule=event_schedule if event_schedule is not None else os.getenv("GITHUB_EVENT_SCHEDULE","")
    manual=event_name=="workflow_dispatch" and bool(manual_refresh)

    need,need_reason=writer_need(root,now)
    after_start=(local.hour>=5)
    writer_eligible=bool(need and (manual or (event_name=="schedule" and after_start)))

    briefing=load(Path(root)/"docs/briefing/latest.json",{})
    briefing_at=briefing.get("generated_at")
    briefing_age=age_hours(briefing_at,now)
    briefing_today=same_et_day(briefing_at,now)
    briefing_stale_for_writer=(not briefing_today) or briefing_age is None or briefing_age>=1.0
    briefing_routine_due=(not briefing_today) or briefing_age is None or briefing_age>=4.5
    refresh_briefing=manual or (event_name=="schedule" and after_start and
        ((writer_eligible and briefing_stale_for_writer) or briefing_routine_due))

    retry_mlb,retry_reason=mlb_retry_relevant(root,now)
    board=load(Path(root)/"docs/mlb/data/latest.json",{})
    board_at=board.get("model_checked_at") or board.get("last_success_at")
    board_age=age_hours(board_at,now)
    board_today=same_et_day(board_at,now)
    board_fresh=bool(board.get("status")=="ready" and board_today and board_age is not None and board_age<1.25)
    refresh_mlb=manual or bool(writer_eligible and retry_mlb and not board_fresh)

    if manual:mode="manual"
    elif writer_eligible and 5<=local.hour<7:mode="morning"
    elif writer_eligible:mode="catch-up"
    elif refresh_briefing:mode="market-refresh"
    else:mode="maintenance"

    result=dict(
        mode=mode,
        event_name=event_name or "unknown",
        event_schedule=event_schedule or "",
        utc_time=now.isoformat(),
        eastern_time=local.isoformat(),
        writer_needed=need,
        writer_reason=need_reason,
        writer_eligible=writer_eligible,
        refresh_briefing=refresh_briefing,
        briefing_at=briefing_at,
        briefing_age_hours=None if briefing_age is None else round(briefing_age,3),
        refresh_mlb=refresh_mlb,
        mlb_retry_relevant=retry_mlb,
        mlb_retry_reason=retry_reason,
        mlb_board_at=board_at,
        mlb_board_age_hours=None if board_age is None else round(board_age,3),
        mlb_board_fresh=board_fresh,
        published_today=len(today_catalog(root,now)),
    )
    return result


def write_outputs(result,path):
    if not path:return
    with open(path,"a") as stream:
        for key in ("mode","writer_eligible","refresh_briefing","refresh_mlb","writer_reason","eastern_time"):
            value=result[key]
            if isinstance(value,bool):value=str(value).lower()
            stream.write(f"{key}={value}\n")


def verify_writer(root=ROOT,now=None,expected=False):
    now=(now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if not expected:
        print(json.dumps(dict(status="not-required")))
        return
    state=today_state(root,now)
    marker=state.get("last_writer_check",{})
    checked=stamp(marker.get("at"))
    if marker.get("status")!="completed" or not checked or abs((now-checked.astimezone(timezone.utc)).total_seconds())>7200:
        raise SystemExit("Writer was expected but today's ledger has no recent completed writer execution marker")
    print(json.dumps(dict(status="verified",last_writer_check=marker,
        slots={k:v.get("status") for k,v in state.get("slots",{}).items()},
        published_today=len(today_catalog(root,now)))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify-writer",action="store_true")
    p.add_argument("--expected-writer",choices=["true","false"],default="false")
    p.add_argument("--event-name",default=os.getenv("GITHUB_EVENT_NAME",""))
    p.add_argument("--event-schedule",default=os.getenv("GITHUB_EVENT_SCHEDULE",""))
    p.add_argument("--manual-refresh",choices=["true","false"],default="false")
    args=p.parse_args()
    if args.verify_writer:
        verify_writer(expected=args.expected_writer=="true")
        return
    result=plan(event_name=args.event_name,event_schedule=args.event_schedule,
        manual_refresh=args.manual_refresh=="true")
    print(json.dumps(result,indent=2))
    write_outputs(result,os.getenv("GITHUB_OUTPUT"))


if __name__=="__main__":main()
