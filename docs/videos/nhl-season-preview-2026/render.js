(async()=>{
  const file=new URLSearchParams(location.search).get('spec')||'timeline.json';
  const spec=await fetch(file).then(r=>r.json());
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
    spec.scenes[i].cards.forEach(([label,value,note],j)=>card(265+j*195,label,value,note,value.includes('−$')?coral:mint));
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
    text('fourthandvalue.com',100,1040,25,mint);fit(spec.footer,940,1040,21,muted,880,500);
    c.fillStyle='#273445';c.fillRect(100,1060,1720,5);c.fillStyle=mint;c.fillRect(100,1060,1720*Math.min(time/spec.duration,1),5);
  }
  draw(0);window.videoReady=true;window.drawVideoFrame=draw;window.videoSpec=spec;
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
