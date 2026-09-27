"""Archive existing authorized NFL game odds and expose exact research offers."""
from nhl.v2.data import digest, iso, write_json


def publish(events, now, root):
    snapshot=dict(ingested_at=iso(now),events=events)
    snapshot_id=digest(snapshot)[:24]
    write_json(root/'data/nfl/lines/raw'/f'{snapshot_id}.json',snapshot)
    rows=[]
    for event in events:
        for book in event.get('bookmakers',[]):
            for market in book.get('markets',[]):
                if market['key'] not in ('h2h','spreads','totals'): continue
                for outcome in market.get('outcomes',[]):
                    rows.append(dict(sport='NFL',event_id=event['id'],game_id=event['id'],
                        game=event['away_team']+' @ '+event['home_team'],home_team=event['home_team'],away_team=event['away_team'],
                        commence_time=event['commence_time'],market=market['key'],market_std=market['key'],
                        market_label={'h2h':'Moneyline','spreads':'Spread','totals':'Game total'}[market['key']],
                        player='',side=outcome['name'],line=outcome.get('point'),price=outcome['price'],
                        book=book['key'],book_label=book.get('title',book['key']),
                        quoted_at=market.get('last_update') or book.get('last_update'),
                        model_probability=None,model_status='No established probability model for this market'))
    value=dict(schema_version=1,status='ready' if rows else 'waiting_for_markets',
        generated_at=iso(now),snapshot_id=snapshot_id,rows=rows)
    write_json(root/'docs/nfl/data/quotes.json',value)
    return value
