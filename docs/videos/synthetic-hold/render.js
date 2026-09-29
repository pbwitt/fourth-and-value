/* Accurate type and charts stay deterministic; AI supplies narration only. */
(async()=>{
  const specFile=new URLSearchParams(location.search).get('spec')||'timeline.json';
  const spec=await fetch(specFile).then(r=>r.json());
  const silent=new URLSearchParams(location.search).has('silent');
  const canvas=document.querySelector('canvas'), c=canvas.getContext('2d');
  const mint='#6ee7b7',white='#edf2f7',muted='#a8b8cd',blue='#7ba9e6',warn='#ffbd7a',red='#ffaba5',dim='#2a3647';
  const total=String(spec.scenes.length).padStart(2,'0');
  const audioContext=silent?null:new AudioContext();
  const buffers=silent?[]:await Promise.all(spec.scenes.map(async s=>audioContext.decodeAudioData(await fetch(s.audio).then(r=>r.arrayBuffer()))));
  function text(t,x,y,size=40,color=white,weight=500,align='left'){c.font=`${weight} ${size}px Arial`;c.fillStyle=color;c.textAlign=align;c.fillText(t,x,y);c.textAlign='left';}
  function wrap(t,x,y,width,size=40,color=white,lineHeight=53){
    c.font=`500 ${size}px Arial`;let line='';
    for(const word of t.split(' ')){const trial=line?line+' '+word:word;if(c.measureText(trial).width>width&&line){text(line,x,y,size,color);y+=lineHeight;line=word;}else line=trial;}
    if(line)text(line,x,y,size,color);return y+lineHeight;
  }
  function fit(t,x,y,size,color=white,weight=700,maxWidth=864){while(size>20){c.font=`${weight} ${size}px Arial`;if(c.measureText(t).width<=maxWidth)break;size-=1;}text(t,x,y,size,color,weight);}
  function pill(x,y,w,h,color,r=20){c.fillStyle=color;c.beginPath();c.roundRect(x,y,Math.max(w,0),h,r);c.fill();}
  function outline(x,y,w,h,color){c.strokeStyle=color;c.lineWidth=4;c.beginPath();c.roundRect(x,y,w,h,12);c.stroke();}

  // Three-book price table (hypothetical Bills −3 / Dolphins +3).
  function books(highlight){
    const rows=[['Book A','−115','−105','4.50%'],['Book B','−108','−112','4.54%'],['Book C','−110','−110','4.55%']];
    const cols=[108,400,640,972];
    text('BOOK',cols[0],1200,24,muted,600);text('BILLS −3',cols[1],1200,24,muted,600);text('DOLPH +3',cols[2],1200,24,muted,600);text('HOLD',cols[3],1200,24,muted,600,'right');
    rows.forEach((r,j)=>{
      const y=1262+j*62;
      text(r[0],cols[0],y,34,white);
      const bestA=highlight&&j===1,bestB=highlight&&j===0;
      if(bestA)outline(cols[1]-14,y-42,150,58,mint);
      if(bestB)outline(cols[2]-14,y-42,150,58,mint);
      text(r[1],cols[1],y,34,bestA?mint:white,bestA?700:500);
      text(r[2],cols[2],y,34,bestB?mint:white,bestB?700:500);
      text(r[3],cols[3],y,34,muted,500,'right');
    });
  }
  // Horizontal hold bars; the synthetic bar is mint, single books neutral.
  function holdBars(items,max,top){
    items.forEach((it,j)=>{
      const y=top+j*44,w=560*it[1]/max;
      text(it[0],108,y+28,26,it[2]?white:muted,it[2]?700:500);
      pill(330,y+6,w,28,it[2]?mint:dim,8);
      text(it[1].toFixed(2)+'%',972,y+30,26,it[2]?mint:muted,it[2]?700:500,'right');
    });
  }

  function draw(time){
    let i=spec.scenes.findIndex(s=>time<s.start+s.duration);if(i<0)i=spec.scenes.length-1;
    const s=spec.scenes[i],local=Math.max(0,time-s.start),enter=Math.min(1,local/.45);
    c.fillStyle='#0b0e13';c.fillRect(0,0,1080,1920);
    const glow=c.createRadialGradient(920,470,0,920,470,900);glow.addColorStop(0,'#153e34');glow.addColorStop(1,'#0b0e13');c.fillStyle=glow;c.fillRect(0,0,1080,1920);
    c.strokeStyle='#ffffff08';c.lineWidth=2;for(let y=200;y<1640;y+=130){c.beginPath();c.moveTo(100,y);c.lineTo(980,y);c.stroke();}
    text('FOURTH & VALUE',108,160,36,mint,700);
    text(spec.section_label||'BETTING BASICS',108,225,25,muted,600);
    text(`${String(i+1).padStart(2,'0')} / ${total}`,825,160,27,muted);
    c.save();c.globalAlpha=0.45+enter*.55;c.translate(0,24*(1-enter));
    text(s.label,108,390,27,mint,700);
    let y=500;for(const line of s.headline.split('\n')){fit(line,108,y,78);y+=100;}
    pill(108,770,864,260,'#13251f');fit(s.metric,150,914,100,s.visual==='arb'?warn:mint,700,780);
    wrap(s.detail,108,1095,840,36,muted,50);
    const v=s.visual;
    if(v==='books')books(false);
    else if(v==='books-best')books(true);
    else if(v==='implied'){
      const a=.5192307692,b=.5121951220,scale=864/(a+b);
      text('BEST BILLS + BEST DOLPHINS',108,1195,25,muted,600);
      pill(108,1250,a*scale-3,46,mint);pill(108+a*scale+3,1250,b*scale-3,46,blue);
      const hundred=108+864/(a+b);
      c.strokeStyle=warn;c.lineWidth=4;c.beginPath();c.moveTo(hundred,1235);c.lineTo(hundred,1310);c.stroke();
      text('100% = fair',hundred,1228,24,warn,600,'right');text('Total 103.14%',972,1350,26,muted,500,'right');
      text('Bills −108',108,1400,30,muted);text('Dolphins −105',650,1400,30,muted);
    }else if(v==='fair'){
      const a=.5034106412;
      text('AFTER PROPORTIONAL DEVIG',108,1215,25,muted,600);
      pill(108,1250,864*a-3,46,mint);pill(108+864*a+3,1250,864*(1-a)-3,46,blue);
      text('Bills 50.34%',108,1340,30,muted);text('Dolphins 49.66%',972,1340,30,muted,500,'right');
    }else if(v==='real'){
      // Sportsbook names stay off-screen (YouTube gambling-policy safety); the article keeps them.
      holdBars([['Synthetic',3.04,true],['Book 1',3.25],['Book 2',4.13],['Book 3',4.54],['Book 4',4.75]],4.75,1175);
    }else if(v==='arb'){
      const a=.5049504950,b=.4878048780;
      text('BILLS −102 + DOLPHINS +105',108,1215,25,muted,600);
      pill(108,1250,864*a-3,46,mint);pill(108+864*a+3,1250,864*b-3,46,blue);
      c.strokeStyle=warn;c.lineWidth=4;c.beginPath();c.moveTo(972,1235);c.lineTo(972,1310);c.stroke();
      text('99.28%',108+864*(a+b),1350,24,muted,500,'right');text('100%',972,1388,24,warn,600,'right');
    }else if(v==='ev'){
      text('−108',108,1235,30,muted);pill(260,1205,650*2.07/2.07*.85,40,mint);text('+$2.07',972,1236,30,mint,700,'right');
      text('−110',108,1300,30,muted);pill(260,1270,650*1.18/2.07*.85,40,mint);text('+$1.18',972,1301,30,mint,700,'right');
      text('−115',108,1365,30,muted);pill(260,1335,650*.91/2.07*.85,40,red);text('−$0.91',972,1366,30,red,700,'right');
    }else if(v==='close'){
      wrap('Same line only. A negative hold is a price observation, not a guaranteed profit.',108,1250,840,36,muted,50);
    }else if(v==='none'){
      text('110 ÷ 210 = 52.38%  ×2  →  104.76%',108,1270,40,muted);
    }
    c.restore();
    // Short phrase captions; approximate timing within each exact audio segment.
    const words=s.narration.split(/\s+/),chunkSize=9,chunks=[];for(let j=0;j<words.length;j+=chunkSize)chunks.push(words.slice(j,j+chunkSize).join(' '));
    const chunk=Math.min(chunks.length-1,Math.floor(local/Math.max(s.speech_duration||s.duration||1,.1)*chunks.length));
    pill(80,1480,920,210,'#06090de8');wrap(chunks[chunk],120,1550,830,43,white,60);
    text('fourthandvalue.com',108,1810,28,mint);
    c.fillStyle='#273445';c.fillRect(108,1845,864,6);c.fillStyle=mint;c.fillRect(108,1845,864*Math.min(time/spec.duration,1),6);
  }
  draw(1);window.videoReady=true;window.drawVideoFrame=draw;
  window.renderVideo=async()=>{
    if(!silent)await audioContext.resume();
    const dest=silent?null:audioContext.createMediaStreamDestination();
    const stream=canvas.captureStream(30);
    if(dest)dest.stream.getAudioTracks().forEach(t=>stream.addTrack(t));
    const mimeType='video/mp4;codecs=avc1.42001E,mp4a.40.2';
    if(!MediaRecorder.isTypeSupported(mimeType))throw new Error('Chrome MP4 recording is required');
    const recorder=new MediaRecorder(stream,{mimeType,videoBitsPerSecond:1800000,audioBitsPerSecond:80000});
    const chunks=[];recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data);};
    const done=new Promise(resolve=>recorder.onstop=resolve);recorder.start(1000);
    const start=silent?performance.now()/1000:audioContext.currentTime+.12;
    if(!silent)spec.scenes.forEach((s,i)=>{const source=audioContext.createBufferSource();source.buffer=buffers[i];source.connect(dest);source.start(start+s.start);});
    await new Promise(resolve=>{function tick(){const now=silent?performance.now()/1000:audioContext.currentTime;const t=Math.max(0,now-start);draw(t);if(t<spec.duration)requestAnimationFrame(tick);else resolve();}tick();});
    recorder.stop();await done;stream.getTracks().forEach(t=>t.stop());
    const blob=new Blob(chunks,{type:'video/mp4'});window.exportedVideoBlob=blob;window.exportedVideoURL=URL.createObjectURL(blob);
    return new Promise(resolve=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result.split(',')[1]);reader.readAsDataURL(blob);});
  };
})().catch(e=>{console.error(e);throw e;});
