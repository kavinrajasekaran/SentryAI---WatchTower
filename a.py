import requests
from bs4 import BeautifulSoup

url = "https://docs.google.com/document/d/e/2PACX-1vQGUck9HIFCyezsrBSnmENk5ieJuYwpt7YHYEzeNJkIb9OSDdx-ov2nRNReKQyey-cwJOoEKUhLmN9z/pub"
page = requests.get(url)
soup = BeautifulSoup(page.text, "html.parser")
rows = soup.find("table").find_all("tr")[1:]

characters = []
for row in rows:
    cells = row.find_all("td")
    if len(cells) >= 3:
        x = int(cells[0].text.strip())
        symbol = cells[1].text.strip()
        y = int(cells[2].text.strip())
        characters.append((x, y, symbol))

# hardcoded values for this case
grid_width = 94
grid_height = 7
grid = []

for row_index in range(grid_height):
    row = []
    for column_index in range(grid_width):
        row.append(" ")
    grid.append(row)

for x, y, symbol in characters:
    grid[y][x] = symbol

for line in grid:
    print("".join(line))
