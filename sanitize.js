const fs = require('fs')
const path = require('path')

const filePath = process.argv[2]; // file path passed as 2nd argument

if (!filePath) {
  console.error('No file path provided.');
  process.exit(1); // Exit if no file is provided
}

// Check if the file exists
fs.exists(filePath, (exists) => {
  if (!exists) {
    console.error(`File not found: ${filePath}`);
    process.exit(1); // Exit if file does not exist
  } else {
    // Read the content of the JSON file
    fs.readFile(filePath, 'utf8', (err, data) => {
      if (err) {
        console.error('Error reading file:', err);
        process.exit(1);
      } else {
        try {
          const prompt = JSON.parse(data);
          let includeEffectivenessIngs = prompt.include_all_effectiveness_items
          let prof =
            ["Leggings", "Boots"].includes(prompt.type.split("-")[0]) ? "tailoring"
              : ["Chestplate", "Helmet"].includes(prompt.type.split("-")[0]) ? "armouring"
                : ["Necklace", "Ring", "Bracelet"].includes(prompt.type.split("-")[0]) ? "jeweling"
                  : prompt.type.contains("Potion") ? "alchemism"
                    : prompt.type.contains("Food") ? "cooking"
                      : prompt.type.contains("Scroll") ? "scribing"
                        : "";

          fetch("https://api.wynncraft.com/v3/item/database?fullResult").then(res => res.json()).then(fullResult => {
            let result = [];
            Object.entries(fullResult).forEach(([key, value]) => {
              if (value.type == "ingredient") {
                if (includeEffectivenessIngs) {
                  if (!isEffectivenessIng(value) && !prompt.ingredient_whitelist.includes(key) || (isEffectivenessIng(value) && !value.requirements.skills.includes(prof))) {
                    return;
                  }
                } else {
                  if (!prompt.ingredient_whitelist.includes(key)) {
                    return
                  }
                }
                let sanitized = {}
                let identifications = []
                sanitized.name = key
                sanitized.skills = value.requirements.skills
                sanitized.duration_modifier = ifNotUndefined(value, "consumableOnlyIDs", () => getOrDefault(value.consumableOnlyIDs, "duration", 0), 0)
                sanitized.charges_modifier = ifNotUndefined(value, "consumableOnlyIDs", () => getOrDefault(value.consumableOnlyIDs, "charges", 0), 0)
                sanitized.position_modifiers = {
                  left: ifNotUndefined(value, "ingredientPositionModifiers", () => getOrDefault(value.ingredientPositionModifiers, "left", 0), 0),
                  right: ifNotUndefined(value, "ingredientPositionModifiers", () => getOrDefault(value.ingredientPositionModifiers, "right", 0), 0),
                  above: ifNotUndefined(value, "ingredientPositionModifiers", () => getOrDefault(value.ingredientPositionModifiers, "above", 0), 0),
                  under: ifNotUndefined(value, "ingredientPositionModifiers", () => getOrDefault(value.ingredientPositionModifiers, "under", 0), 0),
                  touching: ifNotUndefined(value, "ingredientPositionModifiers", () => getOrDefault(value.ingredientPositionModifiers, "touching", 0), 0),
                  not_touching: ifNotUndefined(value, "ingredientPositionModifiers", () => getOrDefault(value.ingredientPositionModifiers, "notTouching", 0), 0)
                }
                sanitized.requirement_modifiers = [
                  ifNotUndefined(value, "itemOnlyIDs", () => getOrDefault(value.itemOnlyIDs, "strengthRequirement", 0), 0),
                  ifNotUndefined(value, "itemOnlyIDs", () => getOrDefault(value.itemOnlyIDs, "dexterityRequirement", 0), 0),
                  ifNotUndefined(value, "itemOnlyIDs", () => getOrDefault(value.itemOnlyIDs, "intelligenceRequirement", 0), 0),
                  ifNotUndefined(value, "itemOnlyIDs", () => getOrDefault(value.itemOnlyIDs, "defenceRequirement", 0), 0),
                  ifNotUndefined(value, "itemOnlyIDs", () => getOrDefault(value.itemOnlyIDs, "agilityRequirement", 0), 0),
                ]
                sanitized.durability_modifier = ifNotUndefined(value, "itemOnlyIDs", () => getOrDefault(value.itemOnlyIDs, "durabilityModifier", 0), 0) / 1000

                if (value.hasOwnProperty("identifications")) {
                  Object.entries(value.identifications).forEach(([id, val]) => {
                    if (typeof val === 'number') {
                      identifications = [...identifications, {
                        id: id,
                        minimum: val,
                        maximum: val
                      }]
                    } else {
                      identifications = [...identifications, {
                        id: id,
                        minimum: val.min,
                        maximum: val.max
                      }]
                    }
                  })
                }
                sanitized.identifications = identifications
                result = [...result, sanitized]
              }
            })
            const jsonString = JSON.stringify(result, null, 2)
            fs.writeFileSync('sanitized.json', jsonString, 'utf8')
          })

          function getOrDefault(obj, key, def) {
            if (obj.hasOwnProperty(key)) {
              return obj[key]
            }
            return def
          }

          function isEffectivenessIng(target) {
            return target.ingredientPositionModifiers.right != 0
              || target.ingredientPositionModifiers.left != 0
              || target.ingredientPositionModifiers.above != 0
              || target.ingredientPositionModifiers.under != 0
              || target.ingredientPositionModifiers.touching != 0
              || target.ingredientPositionModifiers.notTouching != 0

          }

          function isInvalidArray(target, arr) {
            const validValues = new Set(['woodworking', 'weaponsmithing']);
            for (let value of arr) {
              if (!validValues.has(value)) {
                return false;
              }
            }

            const containsWoodworking = arr.includes('woodworking');
            const containsWeaponsmithing = arr.includes('weaponsmithing');

            return (containsWoodworking && containsWeaponsmithing) || (containsWoodworking || containsWeaponsmithing);
          }

          function ifNotUndefined(obj, key, then, orElse) {
            if (obj[key] != undefined) {
              return then()
            }
            return orElse
          }

        } catch (parseError) {
          console.error('Error parsing JSON:', parseError);
          process.exit(1);
        }
      }
    });
  }
});
