# %% NOTEBOOK CELL
# EXEMPLAR: formula-card
# Read the reported balances and official effective date before interpreting the jump.
event_registry = pd.read_csv(ROOT / "sources/structural_events.csv")
merger_date = pd.Timestamp(event_registry.loc[event_registry.CERT.eq(57083), "Window start"].iloc[0])
merger_row = valid.loc[valid.CERT.eq(57083) & valid.date.eq(pd.Timestamp("2023-09-30"))].iloc[0]
assert merger_row.date < merger_date <= merger_row.target_date
merger_receipt = pd.DataFrame([{
    "CERT": 57083, "Predictor date": merger_row.date, "Effective merger": merger_date,
    "Outcome date": merger_row.target_date, "Starting deposits USD": merger_row.DEPDOM*1000,
    "Next deposits USD": merger_row.next_deposits*1000, "Reported growth (%)": 100*merger_row.growth,
}])
merger_receipt.to_csv(OUT / "merger_timeline.csv", index=False)
fig = go.Figure()
fig.add_trace(go.Scatter(x=[merger_row.date,merger_row.target_date],y=[0,0],mode="lines+markers",
    line=dict(color="#AAB9BC",width=4),marker=dict(size=13,color="#3F6294"),showlegend=False,
    hovertemplate="Report date: %{x|%b %d, %Y}<extra></extra>"))
fig.add_trace(go.Scatter(x=[merger_date],y=[0],mode="markers",marker=dict(size=16,color="#A86223",symbol="diamond"),
    showlegend=False,hovertemplate="Merger effective: %{x|%b %d, %Y}<extra></extra>"))
fig.add_annotation(x=merger_date,y=0,text="OCT 1, 2023<br>Merger effective",showarrow=True,ax=120,ay=-90,
    arrowcolor="#A86223",font=dict(color="#A86223",size=17))
for date,label,anchor in [(merger_row.date,f"SEP 30<br>USD {merger_row.DEPDOM/1000:.2f}M","left"),
                         (merger_row.target_date,f"DEC 31<br>USD {merger_row.next_deposits/1000000:.2f}B","right")]:
    fig.add_annotation(x=date,y=-.25,text=label,showarrow=False,xanchor=anchor,font=dict(size=19,color="#3F6294"))
fig.update_xaxes(range=[merger_row.date-pd.Timedelta(days=5),merger_row.target_date+pd.Timedelta(days=5)],visible=False)
fig.update_yaxes(range=[-.65,1.1],visible=False)
chart(fig,"merger_timeline",f"Reported deposit growth: +{100*merger_row.growth:,.0f}%",
    "What happened between these reports? Plus International, CERT 57083; actual date spacing",height=570)
takeaway("The forecast interval contains a merger",
    "Emigrant merged into Plus International under the Emigrant name on October 1, 2023, "
    "one day after the September report date. New York DFS published this record in November 2024. "
    "It supports a retrospective event annotation. Our five inputs contain no event flag; "
    "that does not establish what public information a person could have found at forecast time. "
    "The timeline supplies structural context without attributing every dollar of growth to the merger.")
