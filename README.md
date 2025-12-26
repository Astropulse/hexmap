# Hex World Generator

<img width="2315" height="1803" alt="map" src="https://github.com/user-attachments/assets/9ac65b0f-871a-4036-a7c8-998f95a8f788" />

This project is a procedural hex based world generator that produces a rendered map image, a JSON map description, and a grayscale topography image. It is designed for board game style maps and works especially well for Catan style resource and number layouts.

The generator builds continents, oceans, mountains, hills, rivers, lakes, climate zones, and biomes using layered noise fields and hex grid algorithms. The output is fully deterministic based on a single seed value.

All example and preview images for this project were created using Retro Diffusion  
https://www.retrodiffusion.ai/

---

## Requirements

- Python 3.10 or newer
- Python packages:
  - numpy
  - pillow

Install dependencies:

    pip install numpy pillow

---

## Folder Structure

- map.py  
  Main generator script.

- Tiles/  
  Folder containing tile images. Filenames must follow this pattern:

    <tile_name>_<number>_tile.png

  Example:

    grass_01_tile.png
    grass_02_tile.png

  Multiple variants per tile type are supported and are chosen deterministically per cell.

---

## How to Run

From the project directory:

    python map.py

This will generate:

- map.png  
  Final rendered hex map using the tile images.

- map.json  
  Machine readable map data including tiles, elevation offsets, and dice numbers.

- topography.png  
  Grayscale height map of land tiles only.

---

## Changing the Seed

The entire world is controlled by a single seed value.

Open map.py and locate the Config dataclass near the top:

    seed: int = 5

Change it to any integer:

    seed: int = 12345

Running the script again with a new seed will produce a completely different world.

---

## Changing the Map Size

Map dimensions are defined by:

    rows: int = 200
    cols: int = 300

- Increasing these values creates a larger map
- Decreasing them creates a smaller map

Very large maps will take longer to generate and render.

---

## Modifying World Generation Parameters

All world generation tuning lives inside the Config dataclass. You do not need to modify any code logic to experiment.

High level areas worth adjusting:

### Land vs Ocean
- sea_level
- continental_scale
- continental_octaves

### Mountains and Hills
- mountain_level
- peaks_strength
- ridge_strength
- hill_level

### Climate
- temp_scale
- temp_octaves
- humid_scale
- humid_octaves
- hot_temp, cold_temp, snow_temp

### Biomes
- Forest, jungle, desert, swamp, wheat, and dunes all have thresholds and noise settings
- Small changes can significantly alter biome distribution

### Rivers and Lakes
- river_count
- river_max_len
- lake_strength
- lake_min_coast_dist

Start with small changes and regenerate often. Many parameters interact in non linear ways.

---

## Notes

- Tile selection is deterministic per cell and seed
- Exactly one tile with number 7 is placed inside the largest connected desert region
- Water tiles do not receive resource numbers
- Elevation displacement is smoothed and clamped to avoid visual artifacts
