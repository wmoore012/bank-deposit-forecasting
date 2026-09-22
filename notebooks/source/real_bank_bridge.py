# %% NOTEBOOK CELL
# EXEMPLAR: purposeful-source-preview
# Choose a reproducible illustration near the median asset size, without looking at future growth.
bridge_source = pd.read_csv(ROOT / "data/fdic_financials_2013_2024.csv")
bridge_saved = pd.read_csv(ROOT / "growth_outputs/masterclass/predictions.csv")
bridge_certificates = bridge_saved.loc[bridge_saved.date.eq("2024-03-31"), "CERT"]
bridge_march = bridge_source.loc[bridge_source.REPDTE.eq(20240331) & bridge_source.CERT.isin(bridge_certificates)].copy()
bridge_march["distance_from_median"] = (bridge_march.ASSET - bridge_march.ASSET.median()).abs()
bridge_bank = bridge_march.sort_values(["distance_from_median", "CERT"]).iloc[0]
bridge_prior = bridge_source.loc[bridge_source.CERT.eq(bridge_bank.CERT) & bridge_source.REPDTE.eq(20231231)].iloc[0]
bridge_other_assets = bridge_bank.ASSET - bridge_bank.CHBAL - bridge_bank.LNLSNET
assert bridge_other_assets >= 0
question_card(title="How does a real bank report become five model inputs?", theme=theme,
    body=f"Meet {bridge_bank.NAME}, CERT {int(bridge_bank.CERT)}, in March 2024. "
    "We chose the eligible report closest to the median asset size, breaking ties by CERT. "
    "That rule uses current assets, without choosing an unusually good or bad future outcome. "
    "The illustration comes from the historical evaluation population; a prospective trial needs its own scoring-time eligibility rules.")

# %% NOTEBOOK CELL
# Source balances are in thousands of US dollars. Divide by 1,000 to show millions.
bridge_assets = pd.DataFrame({
    "Asset": ["Cash and balances due", "Net loans and leases", "Other assets (remainder)"],
    "USD million": [bridge_bank.CHBAL/1000, bridge_bank.LNLSNET/1000, bridge_other_assets/1000],
})
fig = go.Figure()
for (_, item), color in zip(bridge_assets.iterrows(), ["#3F6294", "#A8BBC8", "#D9DFE3"]):
    fig.add_trace(go.Bar(x=[item["USD million"]], y=["Reported assets"], orientation="h",
        name=item.Asset, marker_color=color,
        hovertemplate=item.Asset+": USD %{x:.3f}M<extra></extra>"))
fig.update_layout(barmode="stack")
fig.update_xaxes(title="USD millions", range=[0, bridge_bank.ASSET/1000])
chart(fig,"real_bank_assets",f"{bridge_bank.NAME}: USD {bridge_bank.ASSET/1000:.2f}M in assets",
    "March 31, 2024; cash, net loans, and the remaining reported assets",height=440)
display(bridge_assets.round(3))
takeaway("The same division turns different-sized banks into comparable ratios",
    f"This report lists USD {bridge_bank.CHBAL/1000:.3f}M in cash and balances due and USD {bridge_bank.ASSET/1000:.3f}M in assets. "
    f"Divide the first by the second: cash/assets is {bridge_bank.CHBAL/bridge_bank.ASSET:.2%}. "
    "Other assets matter: real balance sheets contain more than cash and loans. "
    "CHBAL is a reported accounting category, not a complete assessment of immediately usable liquidity.")

# %% NOTEBOOK CELL
# Map the actual report to exactly the five original model features.
bridge_inputs = pd.DataFrame([
    ["Deposit size", "log(DEPDOM in USD thousands)", np.log(bridge_bank.DEPDOM)],
    ["Cash / assets", f"{bridge_bank.CHBAL:,} / {bridge_bank.ASSET:,}", bridge_bank.CHBAL/bridge_bank.ASSET],
    ["Loans / assets", f"{bridge_bank.LNLSNET:,} / {bridge_bank.ASSET:,}", bridge_bank.LNLSNET/bridge_bank.ASSET],
    ["Equity / assets", f"{bridge_bank.EQ:,.0f} / {bridge_bank.ASSET:,}", bridge_bank.EQ/bridge_bank.ASSET],
    ["Previous deposit growth", f"log({bridge_bank.DEPDOM:,} / {bridge_prior.DEPDOM:,})", np.log(bridge_bank.DEPDOM/bridge_prior.DEPDOM)],
], columns=["Input", "Calculation from report", "Value before scaling"])
table(bridge_inputs,"Five inputs from this report and its previous quarter",formats={"Value before scaling":"{:.5f}"})
bridge_inputs.to_csv(OUT / "real_bank_inputs.csv",index=False)
bridge_record = bridge_bank.drop(labels=["distance_from_median"]).to_frame().T
bridge_record["Previous deposits"] = bridge_prior.DEPDOM
bridge_record.to_csv(OUT / "real_bank_receipt.csv",index=False)
wm_formula_card(title="What would a hypothetical 10% deposit decline amount to?",theme=theme,items=[
    {"label":"Scenario, not an observed withdrawal", "fallback":f"10% × USD {bridge_bank.DEPDOM/1000:.3f}M deposits = USD {bridge_bank.DEPDOM*.1/1000:.3f}M"},
    {"label":"Compare with the reported cash category", "fallback":f"USD {bridge_bank.DEPDOM*.1/1000:.3f}M / USD {bridge_bank.CHBAL/1000:.3f}M = {bridge_bank.DEPDOM*.1/bridge_bank.CHBAL:.2f} times"},
])
takeaway("That comparison raises a funding question",
    "The hypothetical decline is larger than this report's cash category. Whether the bank could meet payments would depend on timing, asset sales, borrowing capacity, other funding, and the reason deposits changed. "
    "A quarterly deposit change is not a record of cash withdrawals. The model forecasts deposit growth; it does not run this hypothetical cash test or diagnose the bank.")
