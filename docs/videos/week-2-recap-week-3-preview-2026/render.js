(async()=>{
  const spec=await fetch('timeline.json').then(r=>r.json());
  const canvas=document.querySelector('canvas'),c=canvas.getContext('2d');
  const ac=new AudioContext();
  const buffers=await Promise.all(spec.scenes.map(async s=>ac.decodeAudioData(await fetch(s.audio).then(r=>r.arrayBuffer()))));
  const mint='#7ce2bd',white='#edf2f7',muted='#b8c5d6',coral='#f4ae91';
  const text=(t,x,y,size=32,color=white,weight=500)=>{c.font=`${weight} ${size}px Arial`;c.fillStyle=color;c.fillText(t,x,y);};
  function fit(t,x,y,size=48,color=white,width=760,weight=700){while(size>18){c.font=`${weight} ${size}px Arial`;if(c.measureText(t).width<=width)break;size--;}text(t,x,y,size,color,weight);}
  function wrap(t,x,y,width,size=30,color=muted,lh=42){c.font=`500 ${size}px Arial`;let line='';for(const word of t.split(/\s+/)){const next=line?line+' '+word:word;if(c.measureText(next).width>width&&line){text(line,x,y,size,color);y+=lh;line=word;}else line=next;}if(line)text(line,x,y,size,color);return y+lh;}
  function box(x,y,w,h,fill='#13251f'){c.fillStyle=fill;c.beginPath();c.roundRect(x,y,w,h,18);c.fill();}
  function card(y,label,value,note,color=mint){box(1070,y,750,155,'#12212c');text(label,1100,y+36,24,muted,600);fit(value,1100,y+89,44,color,690);text(note,1100,y+128,23,muted);}
  function bars(title,rows,max=100,suffix='%'){
    fit(title,1070,280,32,mint,750);rows.forEach(([label,value,annotation],j)=>{const y=345+j*110;text(label,1070,y,28,white);box(1070,y+20,570,24,'#273445');box(1070,y+20,570*Math.abs(value)/max,24,value<0?coral:mint);text(annotation||`${value}${suffix}`,1665,y+42,28,value<0?coral:mint,700);});
  }
  function visual(i){
    if(i===0){card(260,'WEEK 2','10 unders · 6 overs','What happened at the recorded close');card(440,'WEEK 3','Price the new matchup','Injuries • roles • exact lines');card(620,'OUR COMMITMENT','Grade the misses, too','Every result is in the full article');}
    else if(i===1){bars('POINTS SCORED / CLOSING TOTAL',[['MIN–CHI',12,'12 / 46.5'],['CIN–HOU',26,'26 / 45.5'],['IND–KC',63,'63 / 46.5'],['DET–BUF',72,'72 / 54.5']],80);}
    else if(i===2){card(275,'WEEK 1 UNDERS','7 wins · 9 losses','Recorded closing totals');card(465,'WEEK 2 UNDERS','10 wins · 6 losses','A reversal, not a permanent rule');card(655,'WEEK 2 FAVORITES','11–5 SU · 8–8 ATS','Winning the game is not covering');}
    else if(i===3){bars('UNDER RATE AT REPRESENTATIVE LINES',[['Passing attempts',68.75,'22 / 32'],['Rushing yards',63.33,'57 / 90'],['Rushing attempts',62.07,'36 / 58'],['Receptions',50.30,'84 / 167']]);}
    else if(i===4){bars('NET UNITS BY MODEL MARKET',[['Rushing yards',23.56,'+23.56'],['Pass completions',4.14,'+4.14'],['Receiving yards',-4.40,'−4.40'],['Receptions',-15.20,'−15.20']],25,'u');}
    else if(i===5){card(300,'JONES UNDER 31.5 ATTEMPTS','31 attempts → WIN','DraftKings −110 · +0.91u');card(495,'SKATTEBO UNDER 1.5 CATCHES','4 receptions → LOSS','BetOnline +118 · −1.00u',coral);card(690,'COMBINED FEATURED PROPS','−0.09 units','One unit risked per pick',coral);}
    else if(i===6){card(275,'CAPTURED MARKET','42.5 → 39.5','First captured quote ≠ true opener');card(465,'PROVISIONAL INJURY LAYER','43.6 points','Not a validated causal estimate');card(655,'FINAL SCORE','Philadelphia 24 · Tennessee 20','44 points · Over 39.5 won');}
    else if(i===7){card(280,'MODEL HISTORY','Through September 21','Week 2 results included');card(475,'AUTOMATED INJURY FILE','0 rows','Missing coverage, not zero impact',coral);card(670,'EDITORIAL CHECK','Dated team reports','Linked beside each claim in the article');}
    else if(i===8){card(280,'KRAFT UNDER 3.5','Model 57.9%','Same-line consensus 42.4%');card(475,'DRAFTKINGS','+123','Break-even probability 44.8%');card(670,'BEST LISTED PAYOUT','BetOnline +125','September 22 snapshot • price can move');}
    else if(i===9){card(280,'EARLY RECEPTION RESULTS','3 catches → 2 catches','Two games do not define a role');card(475,'REED: NECK','Estimated DNP Monday','Not a final game designation',coral);card(670,'CONDITIONAL ONLY','3.5 at +120 or better','Recheck role • pass if assumptions fail');}
    else if(i===10){card(280,'WASHINGTON','Marcus Mariota starts','Jayden Daniels sidelined');card(475,'SEATTLE','Darnold practice hoped for','A return to practice ≠ a confirmed start');card(670,'OUR POSITION','Pass on the raw over gap','Correct the matchup before pricing it',coral);}
    else if(i===11){card(320,'MINNESOTA AT TAMPA BAY','Raw 46.1 · Market 43.0','Murray cleared • over research');card(545,'CHARGERS AT BUFFALO','Raw 44.1 · Market 50.5','Combined output • under research');}
    else if(i===12){card(280,'BETRIVERS','Under 50.5 at −114','At 50 points: win');card(475,'BETMGM','Under 50 at −108','At 50 points: refund');card(670,'THE QUESTION','What is the half-point worth?','Compare the number and the price');}
    else if(i===13){card(280,'EAGLES AT BEARS','Williams: week-to-week','Starter changes the interpretation');card(475,'TITANS AT GIANTS','Check Dart availability','Reassess targets and game script');card(670,'TEXANS AT COLTS','Pierce expected to miss weeks','Watch where the routes go');}
    else{card(280,'BEFORE THE BET','Starter → Role → Price','Then challenge the model disagreement');card(475,'FULL ARTICLE + EVIDENCE','fourthandvalue.com','Charts • audit data • sources');card(670,'SUBSCRIBE','@fourthandvalue','Weekly NFL market reviews and previews');}
  }
  function draw(time){
    let i=spec.scenes.findIndex(s=>time<s.start+s.duration);if(i<0)i=spec.scenes.length-1;
    const s=spec.scenes[i],local=Math.max(0,time-s.start);
    c.fillStyle='#0b0e13';c.fillRect(0,0,1920,1080);
    const g=c.createRadialGradient(1550,200,0,1550,200,1400);g.addColorStop(0,'#153e34');g.addColorStop(1,'#0b0e13');c.fillStyle=g;c.fillRect(0,0,1920,1080);
    c.strokeStyle='#ffffff08';for(let y=190;y<900;y+=110){c.beginPath();c.moveTo(100,y);c.lineTo(1820,y);c.stroke();}
    text('FOURTH & VALUE',100,100,34,mint,700);text(spec.section_label,100,147,22,muted,600);
    text(`${String(i+1).padStart(2,'0')} / ${spec.scenes.length}`,1710,100,26,muted);
    fit(s.label,100,270,26,mint,850);let y=365;for(const line of s.headline.split('\n')){fit(line,100,y,68,white,850);y+=86;}
    box(100,565,850,135);fit(s.metric,130,647,47,mint,790);
    wrap(s.detail,100,760,820,29,muted,42);
    visual(i);
    // Phrase captions are approximate, not forced-alignment subtitles.
    const words=s.narration.split(/\s+/),chunks=[];for(let j=0;j<words.length;j+=12)chunks.push(words.slice(j,j+12).join(' '));
    const at=Math.min(chunks.length-1,Math.floor(local/Math.max(s.speech_duration,.1)*chunks.length));
    box(100,905,1720,90,'#06090df0');fit(chunks[at],135,963,34,white,1650,500);
    text('fourthandvalue.com',100,1040,25,mint);text('Odds snapshot: Sep 22, 2026 · 5:46 AM ET',1165,1040,23,muted);
    c.fillStyle='#273445';c.fillRect(100,1060,1720,5);c.fillStyle=mint;c.fillRect(100,1060,1720*Math.min(time/spec.duration,1),5);
  }
  draw(0);window.videoReady=true;window.drawVideoFrame=draw;
  window.renderVideo=async()=>{
    await ac.resume();const dest=ac.createMediaStreamDestination();const stream=canvas.captureStream(30);dest.stream.getAudioTracks().forEach(t=>stream.addTrack(t));
    const mimeType='video/mp4;codecs=avc1.640028,mp4a.40.2';if(!MediaRecorder.isTypeSupported(mimeType))throw new Error('MP4 support required');
    const recorder=new MediaRecorder(stream,{mimeType,videoBitsPerSecond:1450000,audioBitsPerSecond:128000});const chunks=[];
    recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data);};const done=new Promise(resolve=>recorder.onstop=resolve);recorder.start(1000);
    const start=ac.currentTime+.15;spec.scenes.forEach((s,i)=>{const source=ac.createBufferSource();source.buffer=buffers[i];source.connect(dest);source.start(start+s.start);});
    await new Promise(resolve=>{function tick(){const t=Math.max(0,ac.currentTime-start);draw(t);if(t<spec.duration)requestAnimationFrame(tick);else resolve();}tick();});
    recorder.stop();await done;stream.getTracks().forEach(t=>t.stop());
    const blob=new Blob(chunks,{type:'video/mp4'});window.exportedVideoBlob=blob;window.exportedVideoURL=URL.createObjectURL(blob);
    return new Promise(resolve=>{const r=new FileReader();r.onload=()=>resolve(r.result.split(',')[1]);r.readAsDataURL(blob);});
  };
})().catch(e=>{console.error(e);throw e;});
