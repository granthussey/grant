import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

GENUS = {
    "Lactobacillus": "#6683EB",
    "Blautia": "#CFB7A7",
    "Bacteroides": "#1EFEF2",
    "Romboutsia": "#918075",
    "Anaerostipes": "#CFB7A7",
    "Agathobacter": "#918075",
    "Ruminococcaceae NK4A214 group": "#918075",
    "Erysipelotrichaceae UCG-003": "#F79D28",
    "Selenomonas 3": "#D0D0D0",
    "Coprococcus 3": "#918075",
    "Leptotrichia": "#D0D0D0",
    "Faecalibacterium": "#918075",
    "[Clostridium] innocuum group": "#F79D28",
    "Bacillus": "#6683EB",
    "Ruminococcaceae UCG-005": "#918075",
    "[Eubacterium] coprostanoligenes group": "#918075",
    "Subdoligranulum": "#918075",
    "Christensenellaceae R-7 group": "#918075",
    "Streptococcus": "#8FA831",
    "Erysipelatoclostridium": "#F79D28",
    "Coprococcus 2": "#918075",
    "Roseburia": "#918075",
    "Prevotella 7": "#1EFEF2",
    "Ruminiclostridium 9": "#918075",
    "Dorea": "#918075",
    "Peptostreptococcus": "#918075",
    "Slackia": "#D0D0D0",
    "Tyzzerella 4": "#918075",
    "Neisseria": "#F83333",
    "Clostridium sensu stricto 1": "#918075",
    "Actinomyces": "#A8A8A8",
    "Veillonella": "#A8A8A8",
    "Haemophilus": "#F83333",
    "Porphyromonas": "#1EFEF2",
    "Alloprevotella": "#1EFEF2",
    "Bifidobacterium": "#A8A8A8",
    "Clostridium sensu stricto 15": "#918075",
    "[Eubacterium] hallii group": "#918075",
    "Treponema 2": "#D0D0D0",
    "GCA-900066575": "#918075",
    "Prevotella": "#1EFEF2",
    "Corynebacterium": "#D0D0D0",
    "Fretibacterium": "#D0D0D0",
    "[Eubacterium] ventriosum group": "#918075",
    "Collinsella": "#D0D0D0",
    "Lachnoclostridium": "#CFB7A7",
    "Ruminococcus 1": "#918075",
    "Oscillospira": "#918075",
    "Family XIII AD3011 group": "#918075",
    "Pediococcus": "#6683EB",
    "Lachnoanaerobaculum": "#918075",
    "Aggregatibacter": "#F83333",
    "Oribacterium": "#918075",
    "Legionella": "#F83333",
    "Selenomonas 4": "#D0D0D0",
    "Escherichia-Shigella": "#F83333",
    "Shuttleworthia": "#918075",
    "Hungatella": "#918075",
    "Flexilinea": "#D0D0D0",
    "Alistipes": "#1EFEF2",
    "Anaerofilum": "#918075",
    "Akkermansia": "#911EA3",
    "F0058": "#1EFEF2",
    "Atopobium": "#D0D0D0",
    "Ruminococcaceae UCG-014": "#918075",
    "Peptoniphilus": "#918075",
    "Microbacterium": "#D0D0D0",
    "Friedmanniella": "#D0D0D0",
    "Holdemanella": "#F79D28",
    "Abiotrophia": "#0F2F9D",
    "Candidatus Stoquefichus": "#F79D28",
    "Staphylococcus": "#D8D321",
    "[Ruminococcus] torques group": "#918075",
    "Fusobacterium": "#D0D0D0",
    "[Ruminococcus] gnavus group": "#CFB7A7",
    "Parabacteroides": "#16DDD3",
    "Marvinbryantia": "#918075",
    "Coprobacter": "#1EFEF2",
    "Enterococcus": "#35D73C",
    "Ruminococcus 2": "#CFB7A7",
    "[Eubacterium] brachy group": "#918075",
    "Corynebacterium 1": "#D0D0D0",
    "Epulopiscium": "#918075",
    "Paraprevotella": "#1EFEF2",
    "Faecalitalea": "#F79D28",
    "Brevibacterium": "#D0D0D0",
    "CHKCI001": "#918075",
    "Butyricicoccus": "#918075",
    "Methylobacterium": "#F83333",
    "Citrobacter": "#F83333",
    "Acetanaerobacterium": "#918075",
    "Varibaculum": "#D0D0D0",
    "Megasphaera": "#D0D0D0",
    "Anaerococcus": "#918075",
    "Ruminococcaceae UCG-013": "#918075",
    "Ruminococcaceae UCG-002": "#918075",
    "S5-A14a": "#918075",
    "Fournierella": "#918075",
    "Cloacibacillus": "#D0D0D0",
    "Mogibacterium": "#918075",
    "Ruminiclostridium 5": "#918075",
    "[Eubacterium] oxidoreducens group": "#918075",
    "Erysipelotrichaceae UCG-006": "#F79D28",
    "Desulfovibrio": "#F83333",
    "[Ruminococcus] gauvreauii group": "#918075",
    "Lachnoclostridium 5": "#918075",
    "Kocuria": "#D0D0D0",
    "Sutterella": "#F83333",
    "Eisenbergiella": "#918075",
    "Dialister": "#D0D0D0",
    "Macrococcus": "#6683EB",
    "Lachnospiraceae NK4A136 group": "#918075",
    "Catenibacterium": "#F79D28",
    "Leuconostoc": "#0F2F9D",
    "Fructobacillus": "#6683EB",
    "Prevotella 6": "#1EFEF2",
    "Anaeroplasma": "#D0D0D0",
    "Ezakiella": "#918075",
    "Candidatus Berkiella": "#F83333",
    "Anaeroglobus": "#D0D0D0",
    "Eubacterium": "#918075",
    "Terrisporobacter": "#918075",
    "Fusicatenibacter": "#918075",
    "Oscillibacter": "#918075",
    "Intestinibacter": "#918075",
    "[Eubacterium] eligens group": "#918075",
    "Flavonifractor": "#918075",
    "[Eubacterium] xylanophilum group": "#918075",
    "Eggerthella": "#A8A8A8",
    "Merdibacter": "#F79D28",
    "Sellimonas": "#CFB7A7",
    "Barnesiella": "#1EFEF2",
    "Lachnospira": "#918075",
    "Bilophila": "#AF0000",
    "Butyricimonas": "#1EFEF2",
    "Rothia": "#D0D0D0",
    "Odoribacter": "#1EFEF2",
    "Dielma": "#F79D28",
    "Lachnospiraceae ND3007 group": "#918075",
    "Lactococcus": "#6683EB",
    "Acetitomaculum": "#918075",
    "Sphingomonas": "#F83333",
    "Acidaminococcus": "#D0D0D0",
    "Anaerocolumna": "#918075",
    "Clostridium sensu stricto 16": "#918075",
    "Ruminococcaceae UCG-004": "#918075",
    "Weissella": "#6683EB",
    "Lautropia": "#F83333",
    "Negativibacillus": "#918075",
    "Tyzzerella 3": "#918075",
    "Turicibacter": "#F79D28",
    "[Eubacterium] nodatum group": "#918075",
    "Prevotella 9": "#1EFEF2",
    "Ruminococcaceae UCG-010": "#918075",
    "Succiniclasticum": "#D0D0D0",
    "Mailhella": "#F83333",
    "Catenisphaera": "#F79D28",
    "Butyrivibrio": "#918075",
    "Proteus": "#F83333",
    "Intestinimonas": "#918075",
    "Synergistes": "#D0D0D0",
    "Negativicoccus": "#D0D0D0",
    "Parascardovia": "#D0D0D0",
    "Parasutterella": "#AF0000",
    "Anaerosalibacter": "#918075",
    "Anaerofustis": "#918075",
    "Achromobacter": "#F83333",
    "Dubosiella": "#F79D28",
    "Tyzzerella": "#918075",
    "Sedimentibacter": "#918075",
    "Ruminiclostridium": "#918075",
    "Dysgonomonas": "#1EFEF2",
    "Anaerosporobacter": "#918075",
    "Candidatus Soleaferrea": "#918075",
    "Lachnospiraceae UCG-004": "#918075",
    "Clostridioides": "#918075",
    "Paenibacillus": "#6683EB",
    "UBA1819": "#CFB7A7",
    "Rhodococcus": "#D0D0D0",
    "Robinsoniella": "#918075",
    "Mycoplasma": "#D0D0D0",
    "Phascolarctobacterium": "#D0D0D0",
    "Deinococcus": "#D0D0D0",
    "Truepera": "#D0D0D0",
    "Campylobacter": "#D0D0D0",
    "Candidatus Babela": "#D0D0D0",
    "Sneathia": "#D0D0D0",
    "Fastidiosipila": "#918075",
    "Parvimonas": "#918075",
    "Finegoldia": "#918075",
    "Actinotignum": "#D0D0D0",
    "Actinobaculum": "#D0D0D0",
    "Pseudopropionibacterium": "#D0D0D0",
    "[Eubacterium] fissicatena group": "#918075",
    "Xanthomonas": "#F83333",
    "Clostridium sensu stricto 2": "#918075",
    "Defluviitaleaceae UCG-011": "#918075",
    "Flavobacterium": "#1EFEF2",
    "Herbinix": "#918075",
    "Firmicutes bacterium CAG:345": "#F79D28",
    "Coprococcus 1": "#918075",
    "Clostridium sensu stricto 13": "#918075",
    "Bradyrhizobium": "#F83333",
    "Paraclostridium": "#918075",
    "Lachnospiraceae FCS020 group": "#918075",
    "Lysinibacillus": "#6683EB",
    "Clostridium sensu stricto 6": "#918075",
    "Providencia": "#F83333",
    "Peptococcus": "#918075",
    "Genus gut metagenome": "#1EFEF2",
    "Moryella": "#918075",
    "Aerococcus": "#6683EB",
    "Scardovia": "#D0D0D0",
    "Peptoclostridium": "#918075",
    "Coprobacillus": "#F79D28",
    "Bergeyella": "#1EFEF2",
    "Eikenella": "#F83333",
    "Prevotella 2": "#1EFEF2",
    "Morganella": "#F83333",
    "Stenotrophomonas": "#AF0000",
    "Anaerotruncus": "#918075",
    "Tissierella": "#918075",
    "Solobacterium": "#F79D28",
    "DTU089": "#918075",
    "Pseudomonas": "#AF0000",
    "Asteroleplasma": "#F79D28",
    "GCA-900066755": "#918075",
    "Paraeggerthella": "#D0D0D0",
    "Acinetobacter": "#F83333",
    "Johnsonella": "#918075",
    "Howardella": "#918075",
    "Garciella": "#918075",
    "Catabacter": "#918075",
    "Erysipelotrichaceae UCG-004": "#F79D28",
    "Clostridium sensu stricto 3": "#918075",
    "Gemella": "#6683EB",
    "Pyramidobacter": "#D0D0D0",
    "Megamonas": "#D0D0D0",
    "Gordonibacter": "#D0D0D0",
    "Microvirgula": "#F83333",
    "Ruminiclostridium 6": "#918075",
    "Afipia": "#F83333",
    "Aeriscardovia": "#D0D0D0",
    "Olsenella": "#D0D0D0",
    "Gordonia": "#D0D0D0",
    "Faecalibaculum": "#F79D28",
    "Variovorax": "#F83333",
    "Senegalimassilia": "#D0D0D0",
    "Cellulosilyticum": "#918075",
    "Adlercreutzia": "#D0D0D0",
    "F0332": "#D0D0D0",
    "Lachnospiraceae UCG-002": "#918075",
    "Cryobacterium": "#D0D0D0",
    "Pseudoramibacter": "#918075",
    "Ruminiclostridium 1": "#918075",
    "Candidatus Saccharimonas": "#D0D0D0",
    "Cutibacterium": "#D0D0D0",
    "Rodentibacter": "#F83333",
    "Neochlamydia": "#D0D0D0",
    "Prevotellaceae NK3B31 group": "#1EFEF2",
    "Anoxybacillus": "#6683EB",
    "CAG-352": "#918075",
    "metagenome": "#918075",
    "Ruminococcaceae UCG-008": "#918075",
    "Caproiciproducens": "#918075",
    "Coriobacteriaceae UCG-002": "#D0D0D0",
    "Clostridium sp. strain Z6": "#918075",
    "Gardnerella": "#D0D0D0",
    "Lachnospiraceae UCG-003": "#918075",
    "Murdochiella": "#918075",
    "Lachnospiraceae NC2004 group": "#918075",
    "Lachnospiraceae UCG-001": "#918075",
    "Serratia": "#F83333",
    "Ralstonia": "#F83333",
    "Eggerthia": "#F79D28",
    "Murimonas": "#918075",
    "Erysipelotrichaceae UCG-010": "#F79D28",
    "Holdemania": "#F79D28",
    "Novosphingobium": "#F83333",
    "Victivallis": "#D0D0D0",
    "Lachnospiraceae NK3A20 group": "#918075",
    "Ruminococcaceae UCG-003": "#918075",
    "Lachnospiraceae UCG-008": "#918075",
    "[Eubacterium] saphenum group": "#918075",
    "Lachnospiraceae NK4B4 group": "#918075",
    "Propionibacterium": "#D0D0D0",
    "Stomatobaculum": "#918075",
    "Planctopirus": "#D0D0D0",
    "Cuneatibacter": "#918075",
    "Mycobacterium": "#D0D0D0",
    "Libanicoccus": "#D0D0D0",
    "Succinivibrio": "#F83333",
    "Rikenellaceae RC9 gut group": "#1EFEF2",
    "CAG-873": "#1EFEF2",
    "Enterorhabdus": "#D0D0D0",
    "Brachyspira": "#D0D0D0",
    "Lachnospiraceae UCG-007": "#918075",
    "Kluyvera": "#F83333",
    "[Eubacterium] ruminantium group": "#918075",
    "Hydrogenoanaerobacterium": "#918075",
    "Mitsuokella": "#D0D0D0",
    "A2": "#918075",
    "Anaerovibrio": "#D0D0D0",
    "Lachnospiraceae UCG-006": "#918075",
    "Ileibacterium": "#F79D28",
    "Candidatus Protochlamydia": "#D0D0D0",
    "Z20": "#D0D0D0",
    "Massilia": "#F83333",
    "Streptobacillus": "#D0D0D0",
    "Allisonella": "#D0D0D0",
    "Lactonifactor": "#918075",
    "Rhodopseudomonas": "#F83333",
    "Helcococcus": "#918075",
    "Brevundimonas": "#F83333",
    "[Bacteroides] pectinophilus group": "#918075",
    "Prevotellaceae UCG-001": "#1EFEF2",
    "GCA-900066225": "#918075",
    "ASF356": "#918075",
    "Mycetocola": "#D0D0D0",
    "CAG-56": "#918075",
    "Parvibacter": "#D0D0D0",
    "Ruminococcaceae UCG-009": "#918075",
    "W5053": "#918075",
    "Micrococcus": "#D0D0D0",
    "Hydrotalea": "#1EFEF2",
    "Lacibacter": "#1EFEF2",
    "Schlegelella": "#F83333",
    "Bryobacter": "#D0D0D0",
    "Nocardioides": "#D0D0D0",
    "Aeromicrobium": "#D0D0D0",
    "Rikenella": "#1EFEF2",
    "Gallicola": "#918075",
    "Allorhizobium-Neorhizobium-Pararhizobium-Rhizobium": "#F83333",
    "Arcanobacterium": "#D0D0D0",
    "Sarcina": "#918075",
    "CHKCI002": "#D0D0D0",
    "Curtobacterium": "#D0D0D0",
    "Microvirga": "#F83333",
    "Lachnospiraceae FE2018 group": "#918075",
    "Virgibacillus": "#6683EB",
    "Klebsiella": "#AF0000",
    "Alloscardovia": "#D0D0D0",
    "Trueperella": "#D0D0D0",
    "Faecalicoccus": "#F79D28",
    "Angelakisella": "#918075",
    "Tissierella sp. AIP 285.00": "#918075",
    "Capnocytophaga": "#1EFEF2",
    "UC5-1-2E3": "#918075",
    "Anthococcus": "#6683EB",
    "Clostridium sensu stricto 12": "#918075",
    "Allobaculum": "#F79D28",
    "Lachnoclostridium 10": "#918075",
    "Kandleria": "#F79D28",
    "Granulicatella": "#6683EB",
    "Pseudoclavibacter": "#D0D0D0",
    "Harryflintia": "#918075",
    "Raoultibacter": "#D0D0D0",
    "Enterobacter": "#F83333",
    "Chryseobacterium": "#1EFEF2",
    "Clostridium sensu stricto 18": "#918075",
    "Cryptobacterium": "#D0D0D0",
    "Family XIII UCG-001": "#918075",
    "Blastococcus": "#D0D0D0",
    "Modestobacter": "#D0D0D0",
    "Lawsonella": "#D0D0D0",
    "Blastocatella": "#D0D0D0",
    "Geminicoccus": "#F83333",
    "Spirosoma": "#1EFEF2",
    "Bulleidia": "#F79D28",
    "Helicobacter": "#D0D0D0",
    "Paeniclostridium": "#918075",
    "Genus mouse gut metagenome": "#1EFEF2",
    "Leucobacter": "#D0D0D0",
    "Enorma": "#D0D0D0",
    "Plantibacter": "#D0D0D0",
    "Salmonella": "#F83333",
    "[Acetivibrio] ethanolgignens group": "#918075",
    "Muribaculum": "#1EFEF2",
    "Mucispirillum": "#D0D0D0",
    "Aeromonas": "#F83333",
    "Mobiluncus": "#D0D0D0",
    "Papillibacter": "#918075",
    "Phocaeicola": "#1EFEF2",
    "Selenomonas": "#D0D0D0",
    "Tannerella": "#1EFEF2",
    "Sphaerochaeta": "#D0D0D0",
    "Desulfobulbus": "#F83333",
    "Burkholderia-Caballeronia-Paraburkholderia": "#F83333",
    "Acidipropionibacterium": "#D0D0D0",
    "Dermabacter": "#D0D0D0",
    "Ruminococcaceae UCG-011": "#918075",
    "Delftia": "#F83333",
    "Christensenella": "#918075",
    "Incertae Sedis": "#918075",
    "Bosea": "#F83333",
    "Alicyclobacillus": "#6683EB",
    "Carnobacterium": "#6683EB",
    "Photobacterium": "#F83333",
    "Filifactor": "#918075",
    "Actinobacillus": "#F83333",
    "Phocea": "#918075",
    "Pygmaiobacter": "#918075",
    "Thermus": "#D0D0D0",
    "DNF00809": "#D0D0D0",
    "Dolosigranulum": "#6683EB",
    "Ochrobactrum": "#F83333",
    "Sanguibacter": "#D0D0D0",
    "Geobacillus": "#6683EB",
}

FAMILY = {
    "Lactobacillaceae": "#6683EB",
    "Lachnospiraceae": "#CFB7A7",
    "Bacteroidaceae": "#1EFEF2",
    "Peptostreptococcaceae": "#918075",
    "Ruminococcaceae": "#918075",
    "Erysipelotrichaceae": "#F79D28",
    "Veillonellaceae": "#D0D0D0",
    "Leptotrichiaceae": "#D0D0D0",
    "Bacillaceae": "#6683EB",
    "Christensenellaceae": "#918075",
    "Streptococcaceae": "#8FA831",
    "Prevotellaceae": "#1EFEF2",
    "Eggerthellaceae": "#D0D0D0",
    "Neisseriaceae": "#F83333",
    "Clostridiaceae 1": "#918075",
    "Actinomycetaceae": "#D0D0D0",
    "Pasteurellaceae": "#F83333",
    "Porphyromonadaceae": "#1EFEF2",
    "Bifidobacteriaceae": "#D0D0D0",
    "Spirochaetaceae": "#D0D0D0",
    "Corynebacteriaceae": "#D0D0D0",
    "Synergistaceae": "#D0D0D0",
    "Coriobacteriaceae": "#D0D0D0",
    "Family XIII": "#918075",
    "Legionellaceae": "#F83333",
    "Enterobacteriaceae": "#AF0000",
    "Anaerolineaceae": "#D0D0D0",
    "Rikenellaceae": "#1EFEF2",
    "Akkermansiaceae": "#911EA3",
    "Paludibacteraceae": "#1EFEF2",
    "Atopobiaceae": "#D0D0D0",
    "Family XI": "#918075",
    "Microbacteriaceae": "#D0D0D0",
    "Propionibacteriaceae": "#D0D0D0",
    "Aerococcaceae": "#0F2F9D",
    "Staphylococcaceae": "#D8D321",
    "Fusobacteriaceae": "#D0D0D0",
    "Tannerellaceae": "#16DDD3",
    "Barnesiellaceae": "#1EFEF2",
    "Enterococcaceae": "#35D73C",
    "Brevibacteriaceae": "#D0D0D0",
    "Beijerinckiaceae": "#F83333",
    "Desulfovibrionaceae": "#AF0000",
    "Micrococcaceae": "#D0D0D0",
    "Burkholderiaceae": "#F83333",
    "Leuconostocaceae": "#6683EB",
    "Anaeroplasmataceae": "#D0D0D0",
    "Unknown Family": "#F83333",
    "Eubacteriaceae": "#918075",
    "Marinifilaceae": "#1EFEF2",
    "Sphingomonadaceae": "#F83333",
    "Acidaminococcaceae": "#D0D0D0",
    "Dysgonomonadaceae": "#1EFEF2",
    "Paenibacillaceae": "#6683EB",
    "Nocardiaceae": "#D0D0D0",
    "Mycoplasmataceae": "#D0D0D0",
    "Deinococcaceae": "#D0D0D0",
    "Trueperaceae": "#D0D0D0",
    "Campylobacteraceae": "#D0D0D0",
    "Babeliaceae": "#D0D0D0",
    "Xanthomonadaceae": "#AF0000",
    "Defluviitaleaceae": "#918075",
    "Flavobacteriaceae": "#1EFEF2",
    "Xanthobacteraceae": "#F83333",
    "Planococcaceae": "#6683EB",
    "Peptococcaceae": "#918075",
    "Clostridiales vadinBB60 group": "#918075",
    "Weeksellaceae": "#1EFEF2",
    "Pseudomonadaceae": "#AF0000",
    "Moraxellaceae": "#F83333",
    "Aquaspirillaceae": "#F83333",
    "Saccharimonadaceae": "#D0D0D0",
    "Parachlamydiaceae": "#D0D0D0",
    "Muribaculaceae": "#1EFEF2",
    "Victivallaceae": "#D0D0D0",
    "Schlesneriaceae": "#D0D0D0",
    "Mycobacteriaceae": "#D0D0D0",
    "Succinivibrionaceae": "#F83333",
    "Brachyspiraceae": "#D0D0D0",
    "Oligosphaeraceae": "#D0D0D0",
    "Caulobacteraceae": "#F83333",
    "Chitinophagaceae": "#1EFEF2",
    "Solibacteraceae (Subgroup 3)": "#D0D0D0",
    "Nocardioidaceae": "#D0D0D0",
    "Rhizobiaceae": "#F83333",
    "Carnobacteriaceae": "#6683EB",
    "Coriobacteriales Incertae Sedis": "#D0D0D0",
    "Geodermatophilaceae": "#D0D0D0",
    "Blastocatellaceae": "#D0D0D0",
    "Geminicoccaceae": "#F83333",
    "Spirosomaceae": "#1EFEF2",
    "Helicobacteraceae": "#D0D0D0",
    "Deferribacteraceae": "#D0D0D0",
    "Aeromonadaceae": "#F83333",
    "Bacteroidales Incertae Sedis": "#1EFEF2",
    "Desulfobulbaceae": "#F83333",
    "Dermabacteraceae": "#D0D0D0",
    "Alicyclobacillaceae": "#6683EB",
    "Vibrionaceae": "#F83333",
    "Thermaceae": "#D0D0D0",
    "Sanguibacteraceae": "#D0D0D0",
}

PHYLUM = {
    "Firmicutes": "#FFC0CB",
    "Bacteroidetes": "#16DDD3",
    "Fusobacteria": "#D0D0D0",
    "Actinobacteria": "#D0D0D0",
    "Proteobacteria": "#AF0000",
    "Spirochaetes": "#D0D0D0",
    "Synergistetes": "#D0D0D0",
    "Chloroflexi": "#D0D0D0",
    "Verrucomicrobia": "#911EA3",
    "Tenericutes": "#D0D0D0",
    "Deinococcus-Thermus": "#D0D0D0",
    "Epsilonbacteraeota": "#D0D0D0",
    "Dependentiae": "#D0D0D0",
    "Patescibacteria": "#D0D0D0",
    "Chlamydiae": "#D0D0D0",
    "Lentisphaerae": "#D0D0D0",
    "Planctomycetes": "#D0D0D0",
    "Acidobacteria": "#D0D0D0",
    "Deferribacteres": "#D0D0D0",
}

DF_GENUS = pd.DataFrame(GENUS.values(), index=GENUS.keys(), columns=["HexColor"])
DF_FAMILY = pd.DataFrame(FAMILY.values(), index=FAMILY.keys(), columns=["HexColor"])
DF_PHYLUM = pd.DataFrame(PHYLUM.values(), index=PHYLUM.keys(), columns=["HexColor"])

DF_GENUS.index.name = "Genus"
DF_FAMILY.index.name = "Family"
DF_PHYLUM.index.name = "Phylum"

COLORS_TO_HEX = {
    "BROWN": "#CFB7A7",
    "DARK BROWN": "#918075",
    "BLUE": "#6683EB",
    "DARK BLUE": "#0F2F9D",
    "RED": "#F83333",
    "DARK RED": "#AF0000",
    "GRAY": "#D0D0D0",
    "DARK GRAY": "#A8A8A8",
    "PURPLE": "#CA0BE8",
    "DARK PURPLE": "#911EA3",
    "CYAN": "#1EFEF2",
    "DARK CYAN": "#16DDD3",
    "ORANGE": "#F79D28",
    "GREEN": "#35D73C",
    "DARK GREEN": "#066D28",
    "YELLOW": "#FBF528",
    "DARK YELLOW": "#D8D321",
    "BUG GREEN": "#B8D93F",
    "DARK BUG GREEN": "#8FA831",
    "PINK": "#FFC0CB",
    "WHITE": "#FFFFFF",
}

HEX_TO_COLORS = {value: key for key, value in COLORS_TO_HEX.items()}


def match_colors(tax):
    """takes a tax table and matches colors
    
    First, we match on family.
    Next, we match leftover on phylum.
    If all fails, assign #D0D0D0.

    This is CASE-SENSITIVE ,,,,

    """

    # NAME OF INDEX
    if tax.index.name is None:
        index_col = "index"
    else:
        index_col = tax.index.name

    # MATCH ON GENUS
    matched_genus = tax.reset_index().set_index("Genus").join(DF_GENUS)
    not_processed = matched_genus[matched_genus["HexColor"].isnull()]
    matched_genus = matched_genus.dropna()

    # MATCH ON FAMILY
    matched_family = tax.reset_index().set_index("Family").join(DF_FAMILY)
    not_processed = matched_family[matched_family["HexColor"].isnull()]
    matched_family = matched_family.dropna()

    # MATCH ON PHYLUM
    matched_phylum = (
        not_processed.reset_index()
        .set_index("Phylum")
        .drop(columns="HexColor")
        .join(DF_PHYLUM)
    )
    not_processed = matched_phylum[matched_phylum["HexColor"].isnull()]
    matched_phylum = matched_phylum.dropna()

    # ASSIGN WHITE TO ALL OTHERS
    not_processed["HexColor"] = "#FFFFFF"

    # CONCAT TOGETHER
    matched_genus = matched_genus.reset_index().set_index(index_col)
    matched_family = matched_family.reset_index().set_index(index_col)
    matched_phylum = matched_phylum.reset_index().set_index(index_col)
    not_processed = not_processed.reset_index().set_index(index_col)

    matched = pd.concat((matched_genus, matched_family, matched_phylum, not_processed))

    return matched


def display_colors():
    _, ax = plt.subplots(figsize=(0.2, 0.2))
    ax.set_title("FAMILY")

    for key in FAMILY.keys():
        fig, ax = plt.subplots(figsize=(0.2, 0.2))
        plt.axis("off")
        plt.title(key)

        sns.palplot([FAMILY[key]])

    _, ax = plt.subplots(figsize=(0.2, 0.2))
    ax.set_title("PHYLUM")

    for key in PHYLUM.keys():
        fig, ax = plt.subplots(figsize=(0.2, 0.2))
        plt.axis("off")
        plt.title(key)

        sns.palplot([PHYLUM[key]])

    _, ax = plt.subplots(figsize=(0.2, 0.2))
    ax.set_title("ELSE")
    sns.palplot(["#D0D0D0"])


def display_result_PF(matched_tax):
    """run on matched_tax, or the output of match_colors(), to see the assigned colors"""

    matched_tax["PF"] = matched_tax["Phylum"] + "_" + matched_tax["Family"]
    unique_combos = np.sort(matched_tax["PF"].unique())

    for pf in unique_combos:
        fig, ax = plt.subplots(figsize=(0.2, 0.2))
        plt.axis("off")
        plt.title(pf)

        sns.palplot(matched_tax.loc[matched_tax["PF"] == pf, "HexColor"])


def display_result_PFG(matched_tax):
    """run on matched_tax, or the output of match_colors(), to see the assigned colors"""

    matched_tax["PFG"] = (
        matched_tax["Phylum"] + "_" + matched_tax["Family"] + "_" + matched_tax["Genus"]
    )
    unique_combos = np.sort(matched_tax["PFG"].unique())

    for pfg in unique_combos:
        fig, ax = plt.subplots(figsize=(0.2, 0.2))
        plt.axis("off")
        plt.title(pfg)

        sns.palplot(matched_tax.loc[matched_tax["PFG"] == pfg, "HexColor"])


def translate_color(color):
    if "#" in color:
        return HEX_TO_COLORS[color]
    else:
        return COLORS_TO_HEX[color]


def match_on_hct(tax):

    from hctmicrobiomemskcc.dataloading.dataloading import load_microbiome_tables

    HCTDATAPATH = "/Users/granthussey/Lab/Schluter/Archive/data/"

    hctmicrobiometables = load_microbiome_tables(local_dir=HCTDATAPATH)
    hct_taxonomy = hctmicrobiometables[0].join(hctmicrobiometables[1])

    # Match colors on Genus
    tax = tax.join(
        hct_taxonomy.set_index(["Genus"])[["Class", "HexColor"]]
        .groupby("Genus")
        .agg("first"),
        on=["Genus"],
        rsuffix=["hct"],
        how="left",
    )

    # Match colors on Family
    tax.loc[tax.HexColor.isna(), "HexColor"] = (
        tax.loc[tax.HexColor.isna()]
        .join(
            hct_taxonomy.set_index(["Family"])[["Class", "HexColor"]]
            .groupby("Family")
            .agg("first"),
            on=["Family"],
            rsuffix=["hct"],
            how="left",
        )["HexColor['hct']"]
        .values
    )

    # Match colors on Order
    tax.loc[tax.HexColor.isna(), "HexColor"] = (
        tax.loc[tax.HexColor.isna()]
        .join(
            hct_taxonomy.set_index(["Order"])[["Class", "HexColor"]]
            .groupby("Order")
            .agg("first"),
            on=["Order"],
            rsuffix=["hct"],
            how="left",
        )["HexColor['hct']"]
        .values
    )

    # Match colors on Class
    tax.loc[tax.HexColor.isna(), "HexColor"] = (
        tax.loc[tax.HexColor.isna()]
        .join(
            hct_taxonomy.set_index(["Class"])[["HexColor"]]
            .groupby("Class")
            .agg("first"),
            on=["Class"],
            rsuffix=["hct"],
            how="left",
        )["HexColor['hct']"]
        .values
    )

    # Match colors on Phylum
    tax.loc[tax.HexColor.isna(), "HexColor"] = (
        tax.loc[tax.HexColor.isna()]
        .join(
            hct_taxonomy.set_index(["Phylum"])[["HexColor"]]
            .groupby("Phylum")
            .agg("first"),
            on=["Phylum"],
            rsuffix=["hct"],
            how="left",
        )["HexColor['hct']"]
        .values
    )

    # Assign ALL OTHERS as WHITE (#FFFFFF)
    tax.loc[tax["HexColor"].isna(), "HexColor"] = "#FFFFFF"
    tax = tax.drop(columns=[col for col in tax.columns if "hct" in col])

    return tax
