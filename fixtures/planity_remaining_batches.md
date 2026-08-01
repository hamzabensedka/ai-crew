# Planity remaining work — batched autopilot runs

Run **one batch at a time** instead of the full 13-task list. This keeps diffs reviewable and avoids the 50-cycle spin when merges never land.

## Recommended order

1. `planity_remaining_batch_reviews.json` — profile + reviews (3 tasks)
2. `planity_remaining_batch_payment.json` — Stripe + mobile checkout + DevOps (3 tasks)
3. `planity_remaining_batch_provider.json` — provider API + mobile (2 tasks)
4. `planity_remaining_batch_notifications.json` — notifications + BullMQ (2 tasks)
5. `planity_remaining_batch_admin.json` — admin app + review + progress report (3 tasks)

## Example command

```bash
autocrew autopilot \
  --root C:\planity \
  --context output/contexts/planity_clone_context.json \
  --squad output/squads/planity_clone_squad.json \
  --tasks fixtures/planity_remaining_batch_payment.json \
  --max-cycles 5 \
  --build-limit 4 \
  --stagnant-cycles 3 \
  --push \
  --yes
```

## After each batch

- Confirm `Approved` / `Merged` columns are non-zero, or recover branches from `autocrew/...` on origin.
- Fix security blockers before the next batch if the security audit fails.
- Pull latest `master` before starting the next batch.

## Full list

`planity_remaining_tasks.json` remains the complete backlog; use batches for autopilot runs only.
