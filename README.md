# summer26-sustainability-of-agriculture

## Problem Definition:
**Primary question:** Does nitrogen surplus (over-application of nitrogen relative to crop uptake) predict nitrate drinking water violations?

**As a prediction problem:** Given a specified rectangle in the state of Iowa, a value for the average nitrogen surplus in that area, and a date, can we predict the likelihood of observing a nitrate concentation exceeding $10 \text{(mg/L)}$ at specific water sites downstream of the specified area within a month of the the given date?

### Stakeholders: 
- *Environmental regulators (such as local and state environmental/water agencies).* They care about detecting and reducing violations with limited resources.
- *Farmers.* They care about maximizing yield/profits while maintaining sustainable farming practices and complying with local and federal regulations. 
- *Policymakers.* They care about creating effective policy, which requires understanding the impact of agricultural policies on water quality outcomes.
- *Residents.* They care about access to safe drinking water and their exposure risks.
- *Public health officials.* They care about identifying populations which may be disproportionately affected by nitrate contamination. 

### Decisions this analysis will inform:
- *For environmental agencies and regulators:* help identify counties which are at higher risk of nitrate contaminated drinking water, informing allocation of resources for water testing and monitoring.
- *For policymakers:* assist in evaluation of whether policies managing fertilizer use are associated with a reduction in water quality risks to inform future policy decisions.
- *For farmers:* help inform on sustainable farming practices with regards to fertilizer for their communities.
- *For public health officials:* help identify where to prioritize allocation of resources such as water treatment support or public health interventions. 


### Unit of Analysis:
We plan to use county-level data. The time unit will depend on data availability, but potentially we will use **county-year** observations. Water data appears to be available at a higher frequency than data on agricultural nutrient application.

### Scope and boundaries:
Geographically, we will only focus on counties in **Iowa**. The time frame will depend on the data we can collect.

The included features (at the county level) are:
- Agricultural: nitrogen surplus and fertilizer application
- Weather: rainfall amounts
- ET: evapotranspiration data
- Water: nitrate levels

We do not plan to include non-agricultural (such as industrial) causes of nitrate contamination. We will focus on general nitrate surplus in agriculture (from fertilizer, livestock, etc.) or only on fertilizer application based on data we find.

### Anti-goals:
This project will not:
- Prove causation -- it will only ask whether nitrogen surplus is useful in predicting risk of nitrate drinking water violations.
- Evaluate nitrate sources outside of agriculture.


### Possible secondary questions:
How effective is adding different cover crops in off seasons in mitigating nitrogen surplus?

This question can address the following decisions for recommendations to farmers:
- When to plant cover crops
- Which cover crops to use
