# Sample Claims for Testing

This directory contains sample claim files for testing the Manulife SmartClaim Intelligence system.

## Health Insurance Claims

### APPROVE Scenarios (Should be approved based on policy)

- **health_claim_001_approve.txt**: Emergency medical services - broken arm from skiing accident
- **health_claim_002_approve.txt**: Preventive care - routine dental cleaning
- **health_claim_003_approve.txt**: Prescription medication - covered formulary medication

### REJECT Scenarios (Should be rejected based on policy)

- **health_claim_004_reject.txt**: Cosmetic surgery - elective facelift (excluded)
- **health_claim_005_reject.txt**: Pre-existing condition - diabetes diagnosed before policy start
- **health_claim_006_reject.txt**: Extreme sports injury - bungee jumping accident (excluded)

### ESCALATE Scenarios (Ambiguous or insufficient information)

- **health_claim_007_escalate.txt**: Experimental treatment - unclear if covered
- **health_claim_008_escalate.txt**: International medical treatment - unclear coverage
- **health_claim_009_escalate.txt**: Durable medical equipment - unclear if covered

## Travel Insurance Claims

### APPROVE Scenarios (Should be approved based on policy)

- **travel_claim_001_approve.txt**: Medical emergency abroad - food poisoning requiring hospitalization
- **travel_claim_002_approve.txt**: Travel delay - weather-related delay with hotel expenses
- **travel_claim_003_approve.txt**: Lost luggage - airline lost suitcase with documentation

### REJECT Scenarios (Should be rejected based on policy)

- **travel_claim_004_reject.txt**: Trip cancellation - change of mind (excluded)

### ESCALATE Scenarios (Ambiguous or insufficient information)

- **travel_claim_005_escalate.txt**: Trip cancellation - family emergency (needs review)

## Usage

To test a claim, use the CLI:

```bash
# Test a health insurance claim
python -m app.main claim data/claims/health_claim_001_approve.txt --policy-path data/policies/health_policy.txt

# Test a travel insurance claim
python -m app.main claim data/claims/travel_claim_001_approve.txt --policy-path data/policies/travel_policy.txt
```

Note: If the system expects PDFs, you may need to convert the .txt files to PDFs first, or modify the code to accept .txt files directly.
