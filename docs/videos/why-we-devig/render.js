/* Accurate type and charts stay deterministic; AI supplies narration only. */
(async()=>{
  const specFile=new URLSearchParams(location.search).get('spec')||'timeline.json';
  const spec=await fetch(specFile).then(r=>r.json());
  const silent=new URLSearchParams(location.search).has('silent');
  const canvas=document.querySelector('canvas'), c=canvas.getContext('2d');
  const mint='#6ee7b7',white='#edf2f7',muted='#a8b8cd';
  const audioContext=silent?null:new AudioContext();
  const buffers=silent?[]:await Promise.all(spec.scenes.map(async s=>audioContext.decodeAudioData(await fetch(s.audio).then(r=>r.arrayBuffer()))));
  function text(t,x,y,size=40,color=white,weight=500){c.font=`${weight} ${size}px Arial`;c.fillStyle=color;c.fillText(t,x,y);}
  function wrap(t,x,y,width,size=40,color=white,lineHeight=53){
    c.font=`500 ${size}px Arial`;let line='';
    for(const word of t.split(' ')){const trial=line?line+' '+word:word;if(c.measureText(trial).width>width&&line){text(line,x,y,size,color);y+=lineHeight;line=word;}else line=trial;}
    if(line)text(line,x,y,size,color);return y+lineHeight;
  }
  function fit(t,x,y,size,color=white,weight=700,maxWidth=864){while(size>20){c.font=`${weight} ${size}px Arial`;if(c.measureText(t).width<=maxWidth)break;size-=1;}text(t,x,y,size,color,weight);}
  function pill(x,y,w,h,color){c.fillStyle=color;c.beginPath();c.roundRect(x,y,w,h,20);c.fill();}
  function draw(time){
    let i=spec.scenes.findIndex(s=>time<s.start+s.duration);if(i<0)i=spec.scenes.length-1;
    const s=spec.scenes[i],local=Math.max(0,time-s.start),enter=Math.min(1,local/.45);
    c.fillStyle='#0b0e13';c.fillRect(0,0,1080,1920);
    const glow=c.createRadialGradient(920,470,0,920,470,900);glow.addColorStop(0,'#153e34');glow.addColorStop(1,'#0b0e13');c.fillStyle=glow;c.fillRect(0,0,1080,1920);
    c.strokeStyle='#ffffff08';c.lineWidth=2;for(let y=200;y<1640;y+=130){c.beginPath();c.moveTo(100,y);c.lineTo(980,y);c.stroke();}
    text('FOURTH & VALUE',108,160,36,mint,700);
    text(spec.section_label || 'BETTING BASICS',108,225,25,muted,600);
    text(`${String(i+1).padStart(2,'0')} / 08`,825,160,27,muted);
    c.save();c.globalAlpha=0.45+enter*.55;c.translate(0,24*(1-enter));
    text(s.label,108,390,27,mint,700);
    let y=500;for(const line of s.headline.split('\n')){fit(line,108,y,78);y+=100;}
    pill(108,770,864,260,'#13251f');fit(s.metric,150,914,100,mint,700,780);
    wrap(s.detail,108,1095,840,36,muted,50);
    // Two-outcome proportional bars, or signed profit bars for the price example.
    if(i===3||i===4){
      const n=i===3?.5238095238:.5;
      text(i===3?'IMPLIED PROBABILITIES':'AFTER PROPORTIONAL DEVIG',108,1215,25,muted,600);
      pill(108,1250,864*n,46,mint);pill(108+864*n+6,1250,864*n-6,46,'#7ba9e6');
      text('Heads',108,1340,30,muted);text('Tails',585,1340,30,muted);
      if(i===3){c.strokeStyle='#ffbd7a';c.lineWidth=4;c.beginPath();c.moveTo(972,1235);c.lineTo(972,1310);c.stroke();text('100%',882,1388,24,'#ffbd7a');}
    }else if(i===1){
      text('WIN',108,1235,28,muted);pill(260,1200,650,44,mint);text('+$100',770,1295,32,mint);
      text('LOSS',108,1350,28,muted);pill(260,1315,715,44,'#ffaba5');text('−$110',827,1410,32,'#ffaba5');
    }else if(i===2){wrap('Illustration: $10 ÷ $220 = 4.55% of stakes retained. Actual results depend on exposure and outcomes.',108,1235,850,32,muted,45);}
    else if(i===6){text('110 ÷ (110 + 100) = 52.38%',108,1270,43,muted);}
    else if(i===5){wrap('Match the player, market and line. A model estimate can be wrong.',108,1250,840,38,muted,52);}
    c.restore();
    // Short phrase captions; approximate timing within each exact audio segment.
    const words=s.narration.split(/\s+/),chunkSize=9,chunks=[];for(let j=0;j<words.length;j+=chunkSize)chunks.push(words.slice(j,j+chunkSize).join(' '));
    const chunk=Math.min(chunks.length-1,Math.floor(local/Math.max(s.speech_duration,.1)*chunks.length));
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
