-- What each pick was decided on (docs/model-improvement-plan.md, Phase 3).
--
-- model_prob     (existing) the model's probability, conditional on no push
-- market_prob    median no-vig probability across books at the bet's exact line
-- final_prob     sigmoid(w*logit(model) + (1-w)*logit(market)), conditional on no push
-- blend_weight   w, the model's weight in that blend (0.25 until fitted per market)
-- expected_value final_prob x decimal(odds taken) - 1, pushes refunded; 0.03 = +3%
-- decision_at    when the pick was decided (the edition or forecast time)
-- timestamp      (existing) when the bet was placed
--
-- All nullable: bets tracked without a model pick leave them empty. Applied to the
-- production project as migration bet_blend_fields. Safe to re-run.

alter table public.bets
  add column if not exists market_prob numeric,
  add column if not exists final_prob numeric,
  add column if not exists blend_weight numeric,
  add column if not exists expected_value numeric,
  add column if not exists decision_at timestamptz;

alter table public.bets drop constraint if exists bets_blend_probabilities_check;
alter table public.bets add constraint bets_blend_probabilities_check check (
  (market_prob is null or market_prob between 0 and 1)
  and (final_prob is null or final_prob between 0 and 1)
  and (blend_weight is null or blend_weight between 0 and 1));
