import plotly.graph_objects as go

# Coordinates for each continent (approximate center points)
continent_coords = {
    'AF': (1.65, 17.3),
    'AS': (34.05, 100.6),
    'EU': (54.5, 15.3),
    'NA': (54.5, -105.3),
    'OC': (-22.7, 140.0),
    'SA': (-8.78, -55.49),
}

# Pairwise comparison results: (Group 1, Group 2, Mean Difference, Adjusted p-value)
data = [
    ('AF', 'AS', 3688.7, 0.0000),
    ('AF', 'EU', -729.1, 0.0000),
    ('AF', 'NA', -754.3, 0.0000),
    ('AF', 'OC', 3170.9, 0.0000),
    ('AF', 'SA', 3002.7, 0.0000),
    ('AS', 'EU', -4417.8, 0.0000),
    ('AS', 'NA', -4443.0, 0.0000),
    ('AS', 'OC', -517.8, 0.0080),
    ('AS', 'SA', -686.1, 0.0000),
    ('EU', 'OC', 3900.0, 0.0000),
    ('EU', 'SA', 3731.7, 0.0000),
    ('NA', 'OC', 3925.2, 0.0000),
    ('NA', 'SA', 3756.9, 0.0000),
    # Non-significant cases
    ('EU', 'NA', -25.2, 0.9718),
    ('OC', 'SA', -168.3, 0.9352)
]

fig = go.Figure()

for g1, g2, diff, p in data:
    lat1, lon1 = continent_coords[g1]
    lat2, lon2 = continent_coords[g2]
    mid_lat = (lat1 + lat2) / 2
    mid_lon = (lon1 + lon2) / 2

    is_significant = p < 0.05
    color = 'red' if diff > 0 else 'blue'
    width = 2 + abs(diff) / 1000
    dash = 'solid' if is_significant else 'dash'
    color = color if is_significant else 'gray'

    # Draw line between continents
    fig.add_trace(go.Scattergeo(
        lon=[lon1, lon2],
        lat=[lat1, lat2],
        mode='lines',
        line=dict(width=width, color=color, dash=dash),
        opacity=0.7,
        hoverinfo='text',
        text=f"{g1} → {g2}<br>{diff:.1f} g/km<br>p={p:.4f}",
        showlegend=False
    ))

    # Add text label above midpoint
    fig.add_trace(go.Scattergeo(
        lon=[mid_lon],
        lat=[mid_lat + 7],
        mode='text',
        text=[f"{g1}→{g2}: {diff:.0f}g/km"],
        textfont=dict(color=color, size=12),
        hoverinfo='skip',
        showlegend=False
    ))

    # Draw arrival marker
    fig.add_trace(go.Scattergeo(
        lon=[lon2],
        lat=[lat2],
        mode='markers',
        marker=dict(
            size=18,
            symbol='triangle-up',
            color=color,
            line=dict(width=1, color='black')
        ),
        showlegend=False,
        hoverinfo='skip'
    ))

fig.update_layout(
    title='CO₂ Emissions by Departure Continent (Significance Highlighted)',
    geo=dict(
        projection_type='natural earth',
        showland=True,
        landcolor='rgb(245, 245, 245)',
        oceancolor='rgb(220, 235, 255)',
        showocean=True,
        showcountries=True,
        countrycolor='rgb(200, 200, 200)',
        coastlinecolor='rgb(150, 150, 150)',
    ),
    margin=dict(l=0, r=0, t=60, b=0)
)

fig.show()
