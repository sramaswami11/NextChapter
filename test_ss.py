from engines.ss_engine import analyze_claiming_scenarios, get_fra

birth_year = 1960
pia = 2000.0

fra = get_fra(birth_year)
result = analyze_claiming_scenarios(pia, birth_year)

print(f"FRA: {fra} (expect 67.0) {'OK' if fra == 67.0 else 'FAIL'}")

m62 = result["claim_62"]["monthly"]
print(f"Claim 62: ${m62}/mo (expect 1400) {'OK' if m62 == 1400 else 'FAIL'}")

pct62 = result["claim_62"]["pct_vs_fra"]
print(f"  pct_vs_fra: {pct62}% (expect -30.0) {'OK' if pct62 == -30.0 else 'FAIL'}")

mfra = result["claim_fra"]["monthly"]
print(f"Claim FRA: ${mfra}/mo (expect 2000) {'OK' if mfra == 2000 else 'FAIL'}")

m70 = result["claim_70"]["monthly"]
print(f"Claim 70: ${m70}/mo (expect 2480) {'OK' if m70 == 2480 else 'FAIL'}")

pct70 = result["claim_70"]["pct_vs_fra"]
print(f"  pct_vs_fra: {pct70}% (expect 24.0) {'OK' if pct70 == 24.0 else 'FAIL'}")

be_62_fra = result["breakeven_62_vs_fra"]
print(f"Breakeven 62 vs FRA: {be_62_fra} (expect 78.7) {'OK' if be_62_fra == 78.7 else 'FAIL'}")

be_fra_70 = result["breakeven_fra_vs_70"]
print(f"Breakeven FRA vs 70: {be_fra_70} (expect 82.5) {'OK' if be_fra_70 == 82.5 else 'FAIL'}")

print()
print("--- Annual income ---")
print(f"  Claim 62 annual: ${result['claim_62']['annual']:,} (expect 16,800)")
print(f"  Claim FRA annual: ${result['claim_fra']['annual']:,} (expect 24,000)")
print(f"  Claim 70 annual: ${result['claim_70']['annual']:,} (expect 29,760)")

print()
print("--- Zero PIA edge case ---")
r0 = analyze_claiming_scenarios(0.0, 1960)
print(f"  claim_62 monthly: {r0['claim_62']['monthly']} (expect 0) {'OK' if r0['claim_62']['monthly'] == 0 else 'FAIL'}")

print()
print("--- FRA for various birth years ---")
for by, expected in [(1954, 66.0), (1955, 66.1667), (1959, 66.8333), (1960, 67.0)]:
    f = get_fra(by)
    print(f"  born {by}: FRA={f} (expect {expected}) {'OK' if abs(f - expected) < 0.0001 else 'FAIL'}")
