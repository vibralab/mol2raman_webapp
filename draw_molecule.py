from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

smiles = "CCO"

molecule = Chem.MolFromSmiles(smiles)

img = Draw.MolToImage(molecule)

fig, ax = plt.subplots()
ax.plot([0, 1, 2], [0, 1, 4])  # Example plot

# Convert the PIL image to a format compatible with matplotlib
imagebox = OffsetImage(img, zoom=0.2)  # Adjust zoom to fit the image size

# Define the position where the image will be placed (fractional coordinates)
ab = AnnotationBbox(imagebox, (0.8, 0.2), xycoords='axes fraction', frameon=False)

# Add the image to the plot
ax.add_artist(ab)

# Show the plot
plt.show()
