(()=>{
 const $=id=>document.getElementById(id),db=window.supabaseClient;
 let user=null,busy=false;
 const message=text=>{$('message').textContent=text;};
 async function access(){const {data}=await db.auth.getUser();user=data?.user;$('login').hidden=!!user;$('submission').hidden=!user;message(user?'Share your idea below. No title or finished article needed.':'Sign in by email to suggest a story.');}
 $('login-form').onsubmit=async e=>{e.preventDefault();const {error}=await db.auth.signInWithOtp({email:$('email').value.trim(),options:{emailRedirectTo:location.origin+'/editorial/suggest.html',shouldCreateUser:true}});message(error?error.message:'Check your email and open the sign-in link on this device.');};
 $('suggest-form').onsubmit=async e=>{e.preventDefault();if(busy||!user)return;busy=true;$('send').disabled=true;try{const {error}=await db.from('editorial_ideas').insert({user_id:user.id,idea:$('idea').value.trim(),sport:$('sport').value,kind:$('kind').value});if(error)throw error;$('idea').value='';message('Suggestion received. It is private and waiting for editorial review.');}catch(error){message(error.code==='42501'?'Submissions are not enabled yet. Your text is still here; please try again later.':error.message||'Unable to save. Your text is still here.');}finally{busy=false;$('send').disabled=false;}};
 $('signout').onclick=async()=>{await db.auth.signOut();location.reload();};
 db.auth.onAuthStateChange(()=>setTimeout(access,0));access().catch(()=>message('Unable to connect. Please refresh and try again.'));
})();
