const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {PGlite}=require(process.env.FV_PGLITE||'@electric-sql/pglite');
(async()=>{
 const db=new PGlite();
 await db.exec(`create schema auth; create role anon; create role authenticated; create role service_role bypassrls;
 create function auth.jwt() returns jsonb language sql stable as $$select nullif(current_setting('request.jwt.claims',true),'')::jsonb$$;
 grant usage on schema public,auth to anon,authenticated,service_role;`);
 const sql=fs.readFileSync(path.join(__dirname,'../supabase/editorial_diagnostics.sql'),'utf8');await db.exec(sql);await db.exec(sql);
 async function as(role,claims={}){await db.exec('reset role');await db.query("select set_config('request.jwt.claims',$1,false)",[JSON.stringify(claims)]);await db.exec('set role '+role);}
 async function rejected(sql){let failed=false;try{await db.exec(sql);}catch{failed=true;}assert.ok(failed,'Expected permission rejection');}
 await as('service_role');await db.exec(`insert into editorial_pipeline_reports(id,edition_day,observed_at,report) values('run-1','2026-09-26',now(),'{"saved":2}');`);
 await as('anon');await rejected('select * from editorial_pipeline_reports');
 await as('authenticated',{user_metadata:{fv_editor:true},app_metadata:{}});assert.equal((await db.query('select * from editorial_pipeline_reports')).rows.length,0);
 await as('authenticated',{app_metadata:{fv_editor:true}});assert.equal((await db.query('select * from editorial_pipeline_reports')).rows.length,1);
 await rejected("update editorial_pipeline_reports set report='{}'");await rejected('delete from editorial_pipeline_reports');
 await rejected("insert into editorial_pipeline_reports values('fake',current_date,now(),'{}',now())");
 console.log('Diagnostics RLS passed: editors read, service writes, public denied; migration reruns safely.');await db.close();
})().catch(e=>{console.error(e);process.exit(1);});
