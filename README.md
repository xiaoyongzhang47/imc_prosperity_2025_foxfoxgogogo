# IMC Prosperity Competition 2025 Solutions  
**Team foxfoxgogogo**  
- **Hanchen Liu** (Team Captain), Boston College  
- **Xiaoyong Zhang**, Georgia Tech  

---

## Overview  
This repository contains our trading algorithms for the first three rounds of the 2025 IMC Prosperity Competition. We are both physics PhD candidates—Hanchen focused on microstructure research, coding, and the final write-up; Xiaoyong focused on global price-change structure and coding. Although the full competition spans five rounds, we only completed Rounds 1–3; Round 4 code is partially tuned, and Round 5 code is omitted due to time schedule conflicts.

---

## Rounds

### Round 1: Basic Market Making  
**Products introduced:**  
- `RAINFOREST_RESIN` (stable value)  
- `KELP` (oscillating value)  
- `SQUID_INK` (volatile with mean-reversion tendencies)  [oai_citation_attribution:0‡IMC Prosperity](https://imc-prosperity.notion.site/Round-1-19ee8453a09381d18b78cf3c21e5d916)  

**Position limits:**  
- `RAINFOREST_RESIN`: 50  
- `KELP`: 50  
- `SQUID_INK`: 50  [oai_citation_attribution:1‡IMC Prosperity](https://imc-prosperity.notion.site/Round-1-19ee8453a09381d18b78cf3c21e5d916)  

**Approach:**  
- We implemented a two-sided market maker for each product.  
- To mitigate losses from rapid swings in `SQUID_INK`, we employed a moving‐window average to smooth mid-price estimates  [oai_citation_attribution:2‡IMC Prosperity](https://imc-prosperity.notion.site/Round-1-19ee8453a09381d18b78cf3c21e5d916).  
- (Hint from Wiki: large deviations from recent averages tend to revert, providing a statistical edge.)  [oai_citation_attribution:3‡IMC Prosperity](https://imc-prosperity.notion.site/Round-1-19ee8453a09381d18b78cf3c21e5d916)  

---

### Round 2: Basket Trading  
**New products:**  
- `CROISSANTS`  
- `JAMS`  
- `DJEMBES`  
- `PICNIC_BASKET1` (6 × `CROISSANTS`, 3 × `JAMS`, 1 × `DJEMBES`)  
- `PICNIC_BASKET2` (4 × `CROISSANTS`, 2 × `JAMS`)  [oai_citation_attribution:4‡IMC Prosperity](https://imc-prosperity.notion.site/Round-2-19ee8453a09381a580cdf9c0468e9bc8)  

**Position limits:**  
- `CROISSANTS`: 250  
- `JAMS`: 350  
- `DJEMBES`: 60  
- `PICNIC_BASKET1`: 60  
- `PICNIC_BASKET2`: 100  [oai_citation_attribution:5‡IMC Prosperity](https://imc-prosperity.notion.site/Round-2-19ee8453a09381a580cdf9c0468e9bc8)  

**Approach:**  
- We performed linear regression on mid-price data to estimate  
  $$
    P_{\text{basket}} = \alpha + \beta\,P_{\text{base}}
  $$  
- Orders were placed when the basket leg deviated from its fair price as predicted by the regression.

---

### Round 3: Volcanic Rock Options  
**Products introduced:**  
- `VOLCANIC_ROCK` (underlying asset; position limit 400)  
- Five “voucher” options granting the right (but not the obligation) to buy `VOLCANIC_ROCK` at predetermined strikes and expiry; examples:  
  - `VOLCANIC_ROCK_VOUCHER_9500` (Strike = 9 500 SeaShells; Position limit 200; 7 days to expiry)  
  - `VOLCANIC_ROCK_VOUCHER_9750` (Strike = 9 750 SeaShells; Position limit 200; 7 days)  
  - `VOLCANIC_ROCK_VOUCHER_10000` (Strike = 10 000 SeaShells; Position limit 200; 7 days)  [oai_citation_attribution:6‡IMC Prosperity](https://imc-prosperity.notion.site/Round-3-19ee8453a093811082dbcdd1f6c1cd0f)  

**Option pricing & hedging:**  
- We applied the **Black–Scholes** formula to compute each voucher’s fair value  [oai_citation_attribution:7‡Wikipedia](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model?utm_source=chatgpt.com).  
- **Implied volatility** was estimated from the market price via the **volatility smile** methodology  [oai_citation_attribution:8‡Wikipedia](https://en.wikipedia.org/wiki/Volatility_smile?utm_source=chatgpt.com).  
- A **delta-hedge** was maintained by trading `VOLCANIC_ROCK` to neutralize directional exposure  [oai_citation_attribution:9‡Wikipedia](https://en.wikipedia.org/wiki/Delta_neutral?utm_source=chatgpt.com).

---

### Round 4: Luxury Goods Arbitrage *(Partially Tuned)*  
**Product introduced:**  
- `MAGNIFICENT_MACARONS` (limit 75; conversion limit 10)  [oai_citation_attribution:0‡imc-prosperity.notion.site](https://imc-prosperity.notion.site/Round-4-19ee8453a0938112aa5fd7f0d060ffe6)  

**Challenge & Hint:**  
- MACARON prices depend on sunlight, sugar costs, tariffs, and storage.  
- We tried to directly calculate the effective profit of the domestic price and compare it with the market mid‐price to determine whether it is overpriced or underpriced.  [oai_citation_attribution:1‡imc-prosperity.notion.site](https://imc-prosperity.notion.site/Round-4-19ee8453a0938112aa5fd7f0d060ffe6)  
- Our code then flags profitable buy/sell signals, but tuning was limited by time.

---

## Competition Results  
- **Rounds 1–2:** Top 188 globally  
- **Round 3:** Ranked ~400  

---

## Repository Structure  
```
  ├── round1/            # Market maker for basic products
  ├── round2/            # Basket trading algorithms
  ├── round3/            # Volcanic Rock option pricing & hedging
  ├── round4/            # Partially tuned Macarons strategy
  └── README.md          # This write-up
```
---

## References  
- **Prosperity 3 Wiki** (general overview)  [oai_citation_attribution:12‡IMC Prosperity](https://imc-prosperity.notion.site/Prosperity-3-Wiki-19ee8453a09380529731c4e6fb697ea4)  
- **Round 1 products & limits**  [oai_citation_attribution:13‡IMC Prosperity](https://imc-prosperity.notion.site/Round-1-19ee8453a09381d18b78cf3c21e5d916)  
- **Round 2 baskets & limits**  [oai_citation_attribution:14‡IMC Prosperity](https://imc-prosperity.notion.site/Round-2-19ee8453a09381a580cdf9c0468e9bc8)  
- **Round 3 vouchers & mechanics**  [oai_citation_attribution:15‡IMC Prosperity](https://imc-prosperity.notion.site/Round-3-19ee8453a093811082dbcdd1f6c1cd0f)  
- **Round 4 Macarons details**  [oai_citation_attribution:16‡IMC Prosperity](https://imc-prosperity.notion.site/Round-4-19ee8453a0938112aa5fd7f0d060ffe6)  
- **Black–Scholes model**  [oai_citation_attribution:17‡Wikipedia](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model?utm_source=chatgpt.com)  
- **Volatility smile**  [oai_citation_attribution:18‡Wikipedia](https://en.wikipedia.org/wiki/Volatility_smile?utm_source=chatgpt.com)  
- **Delta hedging**  [oai_citation_attribution:19‡Wikipedia](https://en.wikipedia.org/wiki/Delta_neutral?utm_source=chatgpt.com)  
