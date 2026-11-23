LNU IT SUPPORT KNOWLEDGE BASE · internal document · KB-21 · rev. 2026-09
# Processor Thermal Specifications (vendor summary)

## Thermal limits
For Intel Core i5-10400 (Comet Lake, 65 W TDP) the maximum junction temperature TjMax is 100 °C. When a core reaches TjMax the processor asserts PROCHOT# and reduces frequency (thermal throttling). If the temperature keeps rising, THERMTRIP# triggers an immediate shutdown at about 125 °C to protect the silicon. The base frequency is 2.9 GHz and the maximum turbo frequency is 4.3 GHz; sustained turbo depends on the power limits PL1 = 65 W and PL2 = 134 W set by the motherboard.

## Recommended operating range
Idle temperatures between 30 and 45 °C and full-load temperatures below 80 °C are considered healthy with the boxed or tower cooler. Sustained temperatures above 90 °C indicate insufficient cooling: dust, dried thermal interface material or a failed fan. Thermal paste should be reapplied every 3–5 years.
— Page 1 of 1 —
