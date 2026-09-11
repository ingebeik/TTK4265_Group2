bruker Argon, kvikksølv og CO₂ med eksponeringstid 300 ms. Det kan lett endres til 150 ms hvis vi vil, men jeg fant ingen tegn på at de relevante spektrallinjene var overeksponert. Det var bare én mettet piksel i CO₂-bildet.
Alle tre lampene er med i det kombinerte bildet, men jeg bruker bare kvikksølv og argon videre i bølgelengdekalibreringen og FWHM-beregningene. Det er fordi jeg ikke fant noen sikker liste med referansebølgelengder for CO₂, og CO₂ har bredere og mindre tydelige topper.

kan kjøres ved 
python spectral_calibration.py, og alle figurene ligger under Images/results/spectral. Har ikke startet på radiometridelen enda, tenkte å gjøre det imårra hihi


Må/bør kjøres i venv/conda/mamba

python3 -m venv .venv
source .venv/bin/activate 
python -m pip install -r requirements.txt

