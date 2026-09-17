# -*- coding: utf-8 -*-
"""Static classic-Risk board data: territories, adjacency, continents, bonuses."""

TERRITORY_NEIGHBORS = {
    'Alaska': ['Kamchatka', 'Northwest Territory', 'Alberta'],
    'Northwest Territory': ['Alaska', 'Greenland', 'Alberta', 'Ontario'],
    'Greenland': ['Northwest Territory', 'Ontario', 'Quebec', 'Iceland'],
    'Alberta': ['Alaska', 'Northwest Territory', 'Ontario', 'Western United States'],
    'Ontario': ['Northwest Territory', 'Greenland', 'Alberta', 'Quebec', 'Western United States', 'Eastern United States'],
    'Quebec': ['Greenland', 'Ontario', 'Eastern United States'],
    'Western United States': ['Alberta', 'Ontario', 'Eastern United States', 'Central America'],
    'Eastern United States': ['Western United States', 'Ontario', 'Quebec', 'Central America'],
    'Central America': ['Western United States', 'Eastern United States', 'Venezuela'],
    'Venezuela': ['Central America', 'Peru', 'Brazil'],
    'Peru': ['Venezuela', 'Brazil', 'Argentina'],
    'Brazil': ['Peru', 'Venezuela', 'Argentina', 'North Africa'],
    'Argentina': ['Peru', 'Brazil'],
    'Iceland': ['Greenland', 'Great Britain', 'Scandinavia'],
    'Scandinavia': ['Iceland', 'Great Britain', 'Northern Europe', 'Ukraine'],
    'Great Britain': ['Iceland', 'Scandinavia', 'Northern Europe', 'Western Europe'],
    'Northern Europe': ['Scandinavia', 'Great Britain', 'Western Europe', 'Southern Europe', 'Ukraine'],
    'Western Europe': ['Great Britain', 'Northern Europe', 'Southern Europe', 'North Africa'],
    'Southern Europe': ['Western Europe', 'Northern Europe', 'Ukraine', 'Middle East', 'Egypt', 'North Africa'],
    'Ukraine': ['Scandinavia', 'Northern Europe', 'Southern Europe', 'Afghanistan', 'Ural', 'Middle East'],
    'North Africa': ['Brazil', 'Western Europe', 'Southern Europe', 'Egypt', 'East Africa', 'Congo'],
    'Egypt': ['Southern Europe', 'North Africa', 'East Africa', 'Middle East'],
    'East Africa': ['Egypt', 'North Africa', 'Congo', 'South Africa', 'Madagascar', 'Middle East'],
    'Congo': ['North Africa', 'East Africa', 'South Africa'],
    'South Africa': ['Congo', 'East Africa', 'Madagascar'],
    'Madagascar': ['South Africa', 'East Africa'],
    'Middle East': ['Southern Europe', 'Ukraine', 'Afghanistan', 'India', 'East Africa', 'Egypt'],
    'Afghanistan': ['Middle East', 'Ukraine', 'Ural', 'China', 'India'],
    'India': ['Middle East', 'Afghanistan', 'China', 'Siam'],
    'Ural': ['Ukraine', 'Afghanistan', 'China', 'Siberia'],
    'China': ['Siam', 'India', 'Afghanistan', 'Ural', 'Siberia', 'Mongolia'],
    'Siam': ['India', 'China', 'Indonesia'],
    'Siberia': ['Ural', 'China', 'Mongolia', 'Yakutsk', 'Irkutsk'],
    'Mongolia': ['China', 'Siberia', 'Irkutsk', 'Kamchatka', 'Japan'],
    'Irkutsk': ['Mongolia', 'Siberia', 'Yakutsk', 'Kamchatka'],
    'Kamchatka': ['Yakutsk', 'Irkutsk', 'Mongolia', 'Japan', 'Alaska'],
    'Indonesia': ['Siam', 'New Guinea', 'Western Australia'],
    'New Guinea': ['Indonesia', 'Western Australia', 'Eastern Australia'],
    'Western Australia': ['Indonesia', 'New Guinea', 'Eastern Australia'],
    'Eastern Australia': ['Western Australia', 'New Guinea'],
    'Yakutsk': ['Siberia', 'Irkutsk', 'Kamchatka'],
    'Japan': ['Mongolia', 'Kamchatka'],
}

CONTINENTS = {
    'North America': ['Alaska', 'Northwest Territory', 'Greenland', 'Alberta', 'Ontario', 'Quebec', 'Western United States', 'Eastern United States', 'Central America'],
    'South America': ['Venezuela', 'Peru', 'Brazil', 'Argentina'],
    'Europe': ['Iceland', 'Scandinavia', 'Ukraine', 'Great Britain', 'Northern Europe', 'Southern Europe', 'Western Europe'],
    'Africa': ['North Africa', 'Egypt', 'East Africa', 'Congo', 'South Africa', 'Madagascar'],
    'Asia': ['Ural', 'Siberia', 'Yakutsk', 'Kamchatka', 'Afghanistan', 'China', 'Middle East', 'India', 'Siam', 'Japan', 'Irkutsk', 'Mongolia'],
    'Australia': ['Indonesia', 'New Guinea', 'Western Australia', 'Eastern Australia'],
}

CONTINENT_BONUS = {
    'North America': 5,
    'South America': 2,
    'Europe': 5,
    'Africa': 3,
    'Asia': 7,
    'Australia': 2,
}

# Official classic-Risk starting army totals by player count (2-player uses a
# special neutral-army variant that this simulation does not implement).
STARTING_ARMIES = {
    3: 35,
    4: 30,
    5: 25,
    6: 20,
}

TERRITORY_TO_CONTINENT = {
    territory: continent
    for continent, territories in CONTINENTS.items()
    for territory in territories
}

ALL_TERRITORIES = list(TERRITORY_NEIGHBORS.keys())
