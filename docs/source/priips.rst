PRIIPS Performance Statistics
=============================
The EU introduced PRIIPs (Packaged Retail and Insurance-based Investment Products) Regulations in 2014. These regulations require funds to report their risk level and
performance scenarios in a Key Information Document or KID (see https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A02014R1286-20240109). 
The aim of the regulations was to report risk and performance data in a standardised way allowing investors 
to easily compare different investments. 

The regulations separated funds into 4 separate categories. Category 2 funds are funds consisting of linear instruments such as 
equities that have access to a history of returns data. Each fund is supposed to report a best, average, and worst outcome based on the fund's performance history. Daily returns are
used to calculate moments of the returns distribution. These outcomes are reported alongside a stressed outcome based on a worst-case volatility within the data sample. 
Initially, category 2 PRIIPS reported performance forecasts using the `Cornish Fisher <https://en.wikipedia.org/wiki/Cornish%E2%80%93Fisher_expansion>`_ approximation for percentiles based on historical data. The percentiles used were the 10th, 50th, and 90th percentile: 

These calculations were complicated See this `Example <https://www.esma.europa.eu/sites/default/files/library/jc_2017_49_priips_flow_diagram_risk_reward_rev.pdf>`_ Unfortunately, the calculations were also strongly 
procyclical and ended up promising strong, riskless performance at the end of bull markets.

So, after a review in 2021, category 2 funds changed their funds to a simpler calculation (https://www.eiopa.europa.eu/document/download/51861f2c-84a1-4c51-a891-3cd5c47db80c_en?filename=Final%20report%20on%20draft%20RTS%20to%20amend%20PRIIPs%20KID.pdf)
. See https://www.riskquest.com/inspiration/news-events/on-the-road-to-transparency-performance-scenarios-for-priips for some discussion around the calculation. The revision simplified the calculation to the best, worst, and average performance over the 5 year holding period. This required the fund to have at least 10 years of data to calculate the performance. The stressed scenario remained the same after the 
consultation.



Given the problems with the initial PRIIPS calculation, there has now been a divergence between the FCA performance scenarios and 
the EU performance scenarios. The FCA suggests the inclusion of narrative outcomes, while the EU has calculations that should be made. 
The FCA's choice of narrative-based outcomes has now totally lost the point of the PRIIPS performance scenarios - these scenarios 
were supposed to give retail investors a common set of outcomes for comparison between products. Unfortunately, this is no longer the case.
#https://www.simmons-simmons.com/en/publications/cl1au53oy07cf0a92uud6iaxj/the-uk-priips-regime-final-rules-from-1-january-2023
see
https://www.fca.org.uk/publication/policy/ps22-2.pdf - FCA's review of the PRIIPS performance scenarios.

https://www.deloitte.com/lu/en/Industries/investment-management/blogs/priips-rts-calculation-methodology-for-performance-scenarios.html - nice description of the new calculation