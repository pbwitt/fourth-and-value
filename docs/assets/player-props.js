/* Track buttons on the static NHL player pages (scripts/nhl/players.py): each [data-offer]
   cell gets Bet Tracker's shared button for the exact offer embedded in #offers. */
(()=>{
  const data=document.getElementById('offers'),T=window.FVOfferTracker;
  if(!data||!T)return;
  let rows=[];
  try{rows=JSON.parse(data.textContent);}catch{return;}
  document.querySelectorAll('[data-offer]').forEach(cell=>{
    const row=rows[Number(cell.dataset.offer)];
    if(!row)return;
    const button=T.button(row);
    button.setAttribute('aria-label',`Track bet: ${row.player} ${row.side} ${row.line} ${row.market_label} at ${row.book_label||row.book}`);
    cell.append(button);
  });
  T.refresh();
})();
