"""Generate a harder, more diverse eval question set."""

import json
import random
from pathlib import Path


def mutate(seq: str, n: int = 3, seed: int = 42) -> str:
    random.seed(seed)
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    s = list(seq)
    positions = random.sample(range(len(s)), n)
    for pos in positions:
        choices = [a for a in aas if a != s[pos]]
        s[pos] = random.choice(choices)
    return "".join(s)


def main():
    with open("data/processed/proteins.json") as f:
        proteins = {p["id"]: p for p in json.load(f)}

    seqs = {pid: p["sequence"] for pid, p in proteins.items()}

    # Shortcuts
    p53       = seqs["P04637"]
    egfr      = seqs["P00533"]
    hbb       = seqs["P68871"]
    hba       = seqs["P69905"]
    insulin   = seqs["P01308"]
    apoe      = seqs["P02649"]
    sod1      = seqs["P00441"]
    myoglobin = seqs["P02144"]
    prothrombin = seqs["P00734"]
    transferrin = seqs["P02787"]
    brca1     = seqs["P38398"]
    ldha      = seqs["P00338"]
    gapdh     = seqs["P04406"]
    actin     = seqs["P60709"]
    tnf       = seqs["P01375"]
    gr        = seqs["P00390"]

    # Homolog sequences — close to multiple DB members, ambiguous by sequence alone
    mouse_hbb     = seqs["P04444"]   # Mouse HBB-H1  (15 hemoglobins in DB)
    macaque_sod1  = seqs["Q8HXQ0"]  # Rhesus SOD1   (5 SODs in DB, incl. Mn-SODs)
    rat_apoe      = seqs["P0DUY2"]  # Grass rat ApoE (3 ApoE homologs in DB)
    monkey_apoe   = seqs["P0DKW5"]  # Night monkey ApoE
    monkey_hbb    = seqs["P02028"]  # Green monkey HBB
    fly_sod       = seqs["P28755"]  # Fruit fly Cu/Zn SOD
    tarsier_hbd   = seqs["P13558"]  # Tarsier HBD delta-globin
    dunnart_hbb   = seqs["Q28932"]  # Fat-tailed dunnart HBB
    iguana_hbb    = seqs["P18987"]  # Iguana HBB

    questions = []

    # -----------------------------------------------------------------------
    # TEXT-ONLY questions (no sequence) — tests pure text/semantic retrieval
    # -----------------------------------------------------------------------
    text_only = [
        {
            "id": "T001",
            "question": "Which human tumor suppressor protein acts as a transcription factor in response to DNA damage and is mutated in over half of all human cancers?",
            "ground_truth": "TP53 (tumor protein p53, P04637). It is the most frequently mutated gene in human cancer, acting as a sequence-specific transcription factor that triggers cell cycle arrest, DNA repair, or apoptosis upon genotoxic stress.",
            "relevant_uniprot_ids": ["P04637"],
            "category": "function",
            "difficulty": "medium",
            "retrieval_bias": "text",
            "sequence": None,
        },
        {
            "id": "T002",
            "question": "Which receptor tyrosine kinase binds epidermal growth factor and drives cell proliferation in epithelial cancers?",
            "ground_truth": "EGFR (Epidermal growth factor receptor, P00533). It is a transmembrane receptor with an intracellular tyrosine kinase domain; ligand binding triggers autophosphorylation and downstream MAPK/PI3K signaling.",
            "relevant_uniprot_ids": ["P00533"],
            "category": "function",
            "difficulty": "medium",
            "retrieval_bias": "text",
            "sequence": None,
        },
        {
            "id": "T003",
            "question": "What is the primary oxygen-transport protein found in red blood cells of humans?",
            "ground_truth": "Hemoglobin, composed of two alpha (HBA, P69905) and two beta (HBB, P68871) subunits. It binds oxygen cooperatively via heme groups.",
            "relevant_uniprot_ids": ["P68871", "P69905"],
            "category": "function",
            "difficulty": "easy",
            "retrieval_bias": "text",
            "sequence": None,
        },
        {
            "id": "T004",
            "question": "Which pancreatic hormone regulates blood glucose by promoting cellular glucose uptake?",
            "ground_truth": "Insulin (P01308). Produced by pancreatic beta cells, it binds the insulin receptor and triggers GLUT4 translocation, promoting glucose uptake in muscle and adipose tissue.",
            "relevant_uniprot_ids": ["P01308"],
            "category": "function",
            "difficulty": "easy",
            "retrieval_bias": "text",
            "sequence": None,
        },
        {
            "id": "T005",
            "question": "Which human lipoprotein mediates cholesterol transport and has an isoform strongly associated with Alzheimer's disease risk?",
            "ground_truth": "Apolipoprotein E (APOE, P02649). The APOE4 isoform is the strongest known genetic risk factor for late-onset Alzheimer's disease.",
            "relevant_uniprot_ids": ["P02649"],
            "category": "disease",
            "difficulty": "medium",
            "retrieval_bias": "text",
            "sequence": None,
        },
        {
            "id": "T006",
            "question": "Which cytosolic antioxidant enzyme converts superoxide radicals to hydrogen peroxide using copper and zinc cofactors?",
            "ground_truth": "Superoxide dismutase 1 (SOD1, P00441). It catalyzes 2O2- + 2H+ → H2O2 + O2. Mutations in SOD1 cause familial amyotrophic lateral sclerosis (ALS).",
            "relevant_uniprot_ids": ["P00441"],
            "category": "function",
            "difficulty": "medium",
            "retrieval_bias": "text",
            "sequence": None,
        },
        {
            "id": "T007",
            "question": "Which E3 ubiquitin ligase is encoded by a hereditary breast and ovarian cancer susceptibility gene on chromosome 17q?",
            "ground_truth": "BRCA1 (P38398). It functions in DNA double-strand break repair via homologous recombination and has E3 ubiquitin ligase activity through its RING domain.",
            "relevant_uniprot_ids": ["P38398"],
            "category": "function",
            "difficulty": "hard",
            "retrieval_bias": "text",
            "sequence": None,
        },
        {
            "id": "T008",
            "question": "Which serine protease is the zymogen form of thrombin and is activated during the coagulation cascade?",
            "ground_truth": "Prothrombin (coagulation factor II, P00734). Factor Xa cleaves prothrombin to produce thrombin (factor IIa), which then cleaves fibrinogen to form fibrin clots.",
            "relevant_uniprot_ids": ["P00734"],
            "category": "function",
            "difficulty": "medium",
            "retrieval_bias": "text",
            "sequence": None,
        },
        {
            "id": "T009",
            "question": "Which plasma glycoprotein is the main iron-transport protein in human blood?",
            "ground_truth": "Serotransferrin (transferrin, P02787). It binds two Fe3+ ions and delivers iron to cells via receptor-mediated endocytosis of the transferrin receptor complex.",
            "relevant_uniprot_ids": ["P02787"],
            "category": "function",
            "difficulty": "easy",
            "retrieval_bias": "text",
            "sequence": None,
        },
        {
            "id": "T010",
            "question": "Which muscle protein stores and releases oxygen and is responsible for the red color of muscle tissue?",
            "ground_truth": "Myoglobin (P02144). It contains a single heme group and stores oxygen in muscle cells, releasing it during periods of high metabolic demand.",
            "relevant_uniprot_ids": ["P02144"],
            "category": "function",
            "difficulty": "easy",
            "retrieval_bias": "text",
            "sequence": None,
        },
        {
            "id": "T011",
            "question": "What subcellular compartment does BRCA1 primarily localize to, and what nuclear structures does it associate with?",
            "ground_truth": "Nucleus. BRCA1 localizes to the nucleus and forms nuclear foci (BRCA1 nuclear dots) that co-localize with PCNA and sites of DNA repair.",
            "relevant_uniprot_ids": ["P38398"],
            "category": "localization",
            "difficulty": "hard",
            "retrieval_bias": "text",
            "sequence": None,
        },
        {
            "id": "T012",
            "question": "Where is prothrombin synthesized and secreted into?",
            "ground_truth": "Synthesized in the liver (hepatocytes) and secreted into the bloodstream (plasma). It is a vitamin K-dependent plasma protein.",
            "relevant_uniprot_ids": ["P00734"],
            "category": "localization",
            "difficulty": "medium",
            "retrieval_bias": "text",
            "sequence": None,
        },
        {
            "id": "T013",
            "question": "Which point mutation in the beta-globin gene causes the substitution that leads to sickle cell disease?",
            "ground_truth": "p.Glu6Val (E6V) — glutamic acid to valine at position 6 of the mature HBB chain. This creates hemoglobin S (HbS), which polymerizes under low oxygen conditions.",
            "relevant_uniprot_ids": ["P68871"],
            "category": "disease",
            "difficulty": "hard",
            "retrieval_bias": "text",
            "sequence": None,
        },
        {
            "id": "T014",
            "question": "What disease is caused by loss-of-function mutations in the gene encoding superoxide dismutase 1?",
            "ground_truth": "Amyotrophic lateral sclerosis type 1 (ALS1). SOD1 gain-of-toxic-function mutations account for approximately 20% of familial ALS cases.",
            "relevant_uniprot_ids": ["P00441"],
            "category": "disease",
            "difficulty": "medium",
            "retrieval_bias": "text",
            "sequence": None,
        },
        {
            "id": "T015",
            "question": "What inherited syndrome results from germline TP53 mutations?",
            "ground_truth": "Li-Fraumeni syndrome (LFS). Affected individuals develop early-onset tumors including soft-tissue sarcomas, osteosarcomas, breast cancer, brain tumors, and adrenocortical carcinoma.",
            "relevant_uniprot_ids": ["P04637"],
            "category": "disease",
            "difficulty": "medium",
            "retrieval_bias": "text",
            "sequence": None,
        },
        {
            "id": "T016",
            "question": "Which glycolytic enzyme is commonly used as a housekeeping gene in qPCR experiments and is also moonlighting as a transcription factor?",
            "ground_truth": "GAPDH (glyceraldehyde-3-phosphate dehydrogenase, P04406). Beyond glycolysis it has roles in nuclear RNA export, DNA repair, and transcriptional regulation.",
            "relevant_uniprot_ids": ["P04406"],
            "category": "function",
            "difficulty": "hard",
            "retrieval_bias": "text",
            "sequence": None,
        },
        {
            "id": "T017",
            "question": "Which cytokine is also known as cachectin and drives systemic inflammation by activating NF-kB signaling?",
            "ground_truth": "Tumor necrosis factor (TNF-alpha, P01375). It is a pleiotropic cytokine produced mainly by macrophages that mediates inflammation, apoptosis, and immune regulation.",
            "relevant_uniprot_ids": ["P01375"],
            "category": "function",
            "difficulty": "medium",
            "retrieval_bias": "text",
            "sequence": None,
        },
        {
            "id": "T018",
            "question": "What is the primary metabolic role of L-lactate dehydrogenase A in anaerobic conditions?",
            "ground_truth": "LDHA (P00338) catalyzes the conversion of pyruvate to lactate while regenerating NAD+ from NADH, allowing glycolysis to continue under anaerobic conditions.",
            "relevant_uniprot_ids": ["P00338"],
            "category": "function",
            "difficulty": "medium",
            "retrieval_bias": "text",
            "sequence": None,
        },
        {
            "id": "T019",
            "question": "Which structural protein forms the backbone of the cytoskeleton and is one of the most conserved proteins in eukaryotes?",
            "ground_truth": "Beta-actin (ACTB, P60709). It polymerizes into microfilaments and is critical for cell motility, shape, and intracellular transport. It is among the most conserved proteins across eukaryotes.",
            "relevant_uniprot_ids": ["P60709"],
            "category": "function",
            "difficulty": "easy",
            "retrieval_bias": "text",
            "sequence": None,
        },
        {
            "id": "T020",
            "question": "Which FAD-dependent enzyme maintains cellular glutathione in its reduced form to protect against oxidative stress?",
            "ground_truth": "Glutathione reductase (P00390). It reduces glutathione disulfide (GSSG) to GSH using NADPH as the electron donor, maintaining the redox balance of the cell.",
            "relevant_uniprot_ids": ["P00390"],
            "category": "function",
            "difficulty": "medium",
            "retrieval_bias": "text",
            "sequence": None,
        },
        {
            "id": "T021",
            "question": "In what quaternary form does EGFR signal, and what triggers this conformation?",
            "ground_truth": "EGFR signals as a homodimer or heterodimer with other ErbB family members. Ligand binding induces receptor dimerization, which activates the intracellular kinase domain through transphosphorylation.",
            "relevant_uniprot_ids": ["P00533"],
            "category": "structure",
            "difficulty": "hard",
            "retrieval_bias": "text",
            "sequence": None,
        },
        {
            "id": "T022",
            "question": "What cofactor does myoglobin use to bind oxygen, and how does its oxygen affinity compare to hemoglobin?",
            "ground_truth": "Myoglobin uses a heme (iron-porphyrin) cofactor. Its oxygen binding curve is hyperbolic (higher O2 affinity) compared to hemoglobin's sigmoidal curve, reflecting lack of cooperativity.",
            "relevant_uniprot_ids": ["P02144"],
            "category": "structure",
            "difficulty": "hard",
            "retrieval_bias": "text",
            "sequence": None,
        },
        {
            "id": "T023",
            "question": "What vitamin K-dependent modification is required for prothrombin to bind phospholipid membranes during coagulation?",
            "ground_truth": "Gamma-carboxylation of glutamate residues in the Gla domain. Vitamin K-dependent carboxylase converts Glu to gamma-carboxyglutamate (Gca), enabling Ca2+-mediated membrane binding.",
            "relevant_uniprot_ids": ["P00734"],
            "category": "structure",
            "difficulty": "hard",
            "retrieval_bias": "text",
            "sequence": None,
        },
        {
            "id": "T024",
            "question": "Which post-translational modification activates insulin from its precursor form?",
            "ground_truth": "Proteolytic cleavage. Preproinsulin is cleaved to proinsulin, then the C-peptide is removed by prohormone convertases, yielding mature insulin composed of A and B chains linked by disulfide bonds.",
            "relevant_uniprot_ids": ["P01308"],
            "category": "structure",
            "difficulty": "medium",
            "retrieval_bias": "text",
            "sequence": None,
        },
        {
            "id": "T025",
            "question": "What structural domain does BRCA1 use for heterodimerization with BARD1?",
            "ground_truth": "The RING finger domain at the N-terminus of BRCA1. The BRCA1-BARD1 heterodimer has E3 ubiquitin ligase activity and is required for BRCA1 stability and nuclear localization.",
            "relevant_uniprot_ids": ["P38398"],
            "category": "structure",
            "difficulty": "hard",
            "retrieval_bias": "text",
            "sequence": None,
        },
    ]

    # -----------------------------------------------------------------------
    # EXACT SEQUENCE questions — harder, synthesis-level
    # -----------------------------------------------------------------------
    exact_seq = [
        {
            "id": "E001",
            "question": "Based on the provided sequence, what is the biological function of this protein and which disease is it most famously associated with?",
            "ground_truth": "TP53 (P04637) — tumor suppressor and transcription factor. Mutated in >50% of human cancers; germline mutations cause Li-Fraumeni syndrome.",
            "relevant_uniprot_ids": ["P04637"],
            "category": "function",
            "difficulty": "medium",
            "retrieval_bias": "protein",
            "sequence": p53,
        },
        {
            "id": "E002",
            "question": "What enzyme activity does this protein have, and what downstream signaling pathways does it activate upon ligand binding?",
            "ground_truth": "EGFR (P00533) has receptor tyrosine kinase activity. Autophosphorylation activates RAS/MAPK, PI3K/AKT, and JAK/STAT pathways.",
            "relevant_uniprot_ids": ["P00533"],
            "category": "function",
            "difficulty": "hard",
            "retrieval_bias": "protein",
            "sequence": egfr,
        },
        {
            "id": "E003",
            "question": "This sequence is from a heme-containing protein. What is its precise role in oxygen delivery and how does it differ functionally from myoglobin?",
            "ground_truth": "HBB (P68871) is the beta subunit of hemoglobin. Unlike myoglobin, hemoglobin shows cooperative O2 binding (sigmoidal curve), is a tetramer (2α2β), and is the primary O2 carrier in blood.",
            "relevant_uniprot_ids": ["P68871"],
            "category": "function",
            "difficulty": "hard",
            "retrieval_bias": "protein",
            "sequence": hbb,
        },
        {
            "id": "E004",
            "question": "What is the subcellular origin of this protein and what processing steps does it undergo before becoming biologically active?",
            "ground_truth": "Insulin (P01308) is synthesized in pancreatic beta cells as preproinsulin → proinsulin → mature insulin via signal peptide cleavage and C-peptide removal.",
            "relevant_uniprot_ids": ["P01308"],
            "category": "structure",
            "difficulty": "hard",
            "retrieval_bias": "protein",
            "sequence": insulin,
        },
        {
            "id": "E005",
            "question": "What cofactor does this protein require, and what reaction does it catalyze to protect cells from oxidative damage?",
            "ground_truth": "SOD1 (P00441) requires copper (catalytic) and zinc (structural) ions. It catalyzes 2O2•− + 2H+ → H2O2 + O2, dismutating superoxide to hydrogen peroxide.",
            "relevant_uniprot_ids": ["P00441"],
            "category": "function",
            "difficulty": "medium",
            "retrieval_bias": "protein",
            "sequence": sod1,
        },
        {
            "id": "E006",
            "question": "What is the iron-binding capacity of this protein and how does it deliver iron to cells?",
            "ground_truth": "Serotransferrin (P02787) binds two Fe3+ ions. It delivers iron via receptor-mediated endocytosis: the transferrin-receptor complex is internalized, iron released at low pH, and apo-transferrin recycled.",
            "relevant_uniprot_ids": ["P02787"],
            "category": "function",
            "difficulty": "medium",
            "retrieval_bias": "protein",
            "sequence": transferrin,
        },
        {
            "id": "E007",
            "question": "Which coagulation factor activates this protein and what is the immediate product of that activation?",
            "ground_truth": "Factor Xa (with Factor Va, Ca2+, and phospholipid) cleaves prothrombin (P00734) to produce thrombin (factor IIa), which then cleaves fibrinogen to fibrin.",
            "relevant_uniprot_ids": ["P00734"],
            "category": "function",
            "difficulty": "hard",
            "retrieval_bias": "protein",
            "sequence": prothrombin,
        },
        {
            "id": "E008",
            "question": "What is the glycolytic reaction catalyzed by this enzyme and why is it considered a rate-limiting step under anaerobic conditions?",
            "ground_truth": "LDHA (P00338) catalyzes pyruvate + NADH + H+ → lactate + NAD+. It is critical anaerobically because it regenerates NAD+ needed to sustain glycolytic flux when the ETC is unavailable.",
            "relevant_uniprot_ids": ["P00338"],
            "category": "function",
            "difficulty": "medium",
            "retrieval_bias": "protein",
            "sequence": ldha,
        },
        {
            "id": "E009",
            "question": "What is the function of this protein in the context of inflammation, and what receptor does it signal through?",
            "ground_truth": "TNF (P01375) is a proinflammatory cytokine that signals through TNFR1 and TNFR2. It activates NF-κB and MAPK pathways, inducing inflammation, apoptosis, and immune cell activation.",
            "relevant_uniprot_ids": ["P01375"],
            "category": "function",
            "difficulty": "medium",
            "retrieval_bias": "protein",
            "sequence": tnf,
        },
        {
            "id": "E010",
            "question": "Identify this protein and explain how its oxygen-binding properties differ from the related tetramer that circulates in blood.",
            "ground_truth": "Myoglobin (P02144). Unlike hemoglobin, it is monomeric with a single heme group, shows hyperbolic O2 binding (no cooperativity), has higher O2 affinity at physiological pO2, and functions as a muscle O2 store.",
            "relevant_uniprot_ids": ["P02144"],
            "category": "function",
            "difficulty": "hard",
            "retrieval_bias": "protein",
            "sequence": myoglobin,
        },
    ]

    # -----------------------------------------------------------------------
    # HYBRID-CRITICAL questions
    # Sequence alone is ambiguous (multiple close DB matches in the same family).
    # Question text disambiguates toward the specific human target.
    # Neither modality alone reliably retrieves the right protein at rank 1;
    # hybrid succeeds by combining both signals.
    # -----------------------------------------------------------------------
    hybrid_q = [
        # --- Hemoglobin family (15 members in DB) ---
        # Each question uses disease/function terms that appear ONLY in the human
        # protein's annotation, making text retrieval strongly prefer the human target
        # while the non-human sequence makes protein retrieval ambiguous.

        # --- Hemoglobin family (15 members in DB) ---
        {
            "id": "HY001",
            "question": "A beta-globin sequence is provided. Find the human protein associated with Heinz body anemias and non-enzymatic glycation in diabetes mellitus.",
            "ground_truth": "Human HBB (P68871). Unstable HBB variants cause Heinz body anemias (HEIBAN); the N-terminal beta chain undergoes non-enzymatic glycation forming HbA1c, a key diabetes monitoring marker.",
            "relevant_uniprot_ids": ["P68871"],
            "category": "disease",
            "difficulty": "hard",
            "retrieval_bias": "hybrid",
            "sequence": mouse_hbb,
        },
        {
            "id": "HY002",
            "question": "Using this beta-globin sequence as a structural guide, find the human beta-globin whose glycation at its N-terminus by glucose is used clinically to monitor long-term blood glucose control.",
            "ground_truth": "Human HBB (P68871). Glucose reacts non-enzymatically with the beta chain N-terminus forming a ketoamine (HbA1c). HbA1c percentage reflects average blood glucose over the 120-day RBC lifespan and is the standard diabetes monitoring marker.",
            "relevant_uniprot_ids": ["P68871"],
            "category": "function",
            "difficulty": "hard",
            "retrieval_bias": "hybrid",
            "sequence": monkey_hbb,
        },
        {
            "id": "HY003",
            "question": "This globin sequence is from a marsupial. Find the human monomeric heme protein that stores oxygen in muscle and acts as a nitrite reductase and pseudoperoxidase.",
            "ground_truth": "Myoglobin (P02144). It is a monomeric heme protein in muscle cells with nitrite reductase and pseudoperoxidase activities in addition to oxygen storage. Distinct from hemoglobin, which is tetrameric and circulates in blood.",
            "relevant_uniprot_ids": ["P02144"],
            "category": "function",
            "difficulty": "hard",
            "retrieval_bias": "hybrid",
            "sequence": dunnart_hbb,
        },
        {
            "id": "HY004",
            "question": "The provided sequence is a delta-globin. Using it as a structural reference, identify the human beta-globin associated with hereditary hemolytic anemia and congenital dyserythropoietic anemia.",
            "ground_truth": "Human HBB (P68871). Mutations in HBB cause hereditary hemolytic anemias and congenital dyserythropoietic anemia. HBB is the predominant adult beta-like subunit in HbA, unlike the minor delta-globin (HBD) in HbA2.",
            "relevant_uniprot_ids": ["P68871"],
            "category": "disease",
            "difficulty": "hard",
            "retrieval_bias": "hybrid",
            "sequence": tarsier_hbd,
        },
        {
            "id": "HY005",
            "question": "This reptile globin sequence is provided. Find the human beta-globin subunit detected in homozygous alpha-thalassemia as part of hemoglobin Portland-2.",
            "ground_truth": "Human HBB (P68871). In homozygous alpha-thalassemia (Hb Bart's hydrops fetalis), HBB forms hemoglobin Portland-2 (zeta2/beta2) with zeta chains — a clinically significant finding used in diagnosis.",
            "relevant_uniprot_ids": ["P68871"],
            "category": "disease",
            "difficulty": "hard",
            "retrieval_bias": "hybrid",
            "sequence": iguana_hbb,
        },

        # --- SOD family (5 members in DB: 2 Mn-SOD, 2 Cu/Zn non-human, 1 human Cu/Zn) ---
        {
            "id": "HY006",
            "question": "The sequence is from an insect Cu/Zn superoxide dismutase. Find the human Cu/Zn SOD associated with amyotrophic lateral sclerosis and known to also oxidize hydrogen sulfide.",
            "ground_truth": "Human SOD1 (P00441). It dismutates superoxide and also catalyzes H2S oxidation to sulfate. Gain-of-toxic-function mutations cause ALS1 (amyotrophic lateral sclerosis type 1), with pathological aggregation in motor neurons.",
            "relevant_uniprot_ids": ["P00441"],
            "category": "disease",
            "difficulty": "hard",
            "retrieval_bias": "hybrid",
            "sequence": fly_sod,
        },
        {
            "id": "HY007",
            "question": "Using this primate SOD sequence as a structural reference, identify the human homodimeric superoxide dismutase whose pathogenic variants are polyubiquitinated by RNF19A and aggregate in motor neurons.",
            "ground_truth": "Human SOD1 (P00441). Pathogenic ALS1 variants (e.g., Arg-38, Arg-47, Arg-86, Ala-94) are polyubiquitinated by RNF19A/MARCH5 for degradation; they also aggregate in mitochondria of motor neurons.",
            "relevant_uniprot_ids": ["P00441"],
            "category": "disease",
            "difficulty": "hard",
            "retrieval_bias": "hybrid",
            "sequence": macaque_sod1,
        },

        # --- ApoE family (3 ApoE homologs in DB) ---
        {
            "id": "HY008",
            "question": "The provided sequence is an apolipoprotein E from a primate. Find the human ApoE that interacts with amyloid-beta peptide and MAPT, and is linked to hyperlipoproteinemia type 3.",
            "ground_truth": "Human APOE (P02649). It interacts with APP/amyloid-beta and MAPT (tau), linking it to neurodegeneration. APOE2 homozygosity causes hyperlipoproteinemia type 3 (HLPP3) — elevated IDL cholesterol and premature atherosclerosis.",
            "relevant_uniprot_ids": ["P02649"],
            "category": "disease",
            "difficulty": "hard",
            "retrieval_bias": "hybrid",
            "sequence": monkey_apoe,
        },
        {
            "id": "HY009",
            "question": "This ApoE sequence is from a rodent. Find the human apolipoprotein E expressed by astrocytes in the cerebral cortex and associated with Alzheimer disease and amyloidosis.",
            "ground_truth": "Human APOE (P02649). It is the primary apolipoprotein in the CNS, produced by astrocytes. APOE4 increases Alzheimer disease risk; APOE is linked to amyloidosis and interacts with secreted SORL1 in cerebrospinal fluid.",
            "relevant_uniprot_ids": ["P02649"],
            "category": "disease",
            "difficulty": "hard",
            "retrieval_bias": "hybrid",
            "sequence": rat_apoe,
        },

        # --- Mutant sequences with strong disambiguation text ---
        {
            "id": "HY010",
            "question": "This sequence differs by a few residues from a human tumor suppressor. Find the protein that activates CDKN1A and BAX transcription and whose germline mutations cause Li-Fraumeni syndrome.",
            "ground_truth": "TP53 (P04637). It is stabilized by DNA damage and transcriptionally activates CDKN1A (p21, causing G1 arrest) and BAX (apoptosis). Germline mutations cause Li-Fraumeni syndrome with early-onset multi-tissue cancers.",
            "relevant_uniprot_ids": ["P04637"],
            "category": "disease",
            "difficulty": "hard",
            "retrieval_bias": "hybrid",
            "sequence": mutate(p53, n=3, seed=1),
        },
        {
            "id": "HY011",
            "question": "This near-identical sequence belongs to an enzyme that also oxidizes hydrogen sulfide and whose pathogenic variants aggregate in mitochondria, causing amyotrophic lateral sclerosis 1.",
            "ground_truth": "SOD1 (P00441). It has dual activity: superoxide dismutation and H2S oxidation. ALS1 pathogenic variants (e.g., Arg-86, Ala-94) accumulate in mitochondria of motor neurons, causing neurodegeneration.",
            "relevant_uniprot_ids": ["P00441"],
            "category": "disease",
            "difficulty": "hard",
            "retrieval_bias": "hybrid",
            "sequence": mutate(sod1, n=2, seed=7),
        },
        {
            "id": "HY012",
            "question": "The sequence has a few substitutions. Find the human beta-globin whose N-terminus undergoes glucose-mediated glycation forming a ketoamine, and which forms hemoglobin Portland-2 in alpha-thalassemia.",
            "ground_truth": "HBB (P68871). Its N-terminal glycation by glucose forms HbA1c (diabetes monitoring marker). In homozygous alpha-thalassemia it pairs with zeta chains forming hemoglobin Portland-2.",
            "relevant_uniprot_ids": ["P68871"],
            "category": "disease",
            "difficulty": "hard",
            "retrieval_bias": "hybrid",
            "sequence": mutate(hbb, n=2, seed=3),
        },
    ]

    # -----------------------------------------------------------------------
    # COMPARATIVE / MULTI-PROTEIN questions
    # -----------------------------------------------------------------------
    comparative_q = [
        {
            "id": "C001",
            "question": "Compare the oxygen-binding mechanisms of myoglobin and hemoglobin beta subunit. Why does hemoglobin show cooperative binding while myoglobin does not?",
            "ground_truth": "Myoglobin (P02144) is monomeric and shows hyperbolic O2 binding with high affinity. Hemoglobin (HBB P68871 + HBA P69905) is a tetramer showing cooperative (sigmoidal) binding due to allosteric communication between subunits (T→R state transition). Cooperativity allows efficient O2 loading in the lungs and unloading in tissues.",
            "relevant_uniprot_ids": ["P02144", "P68871", "P69905"],
            "category": "comparative",
            "difficulty": "hard",
            "retrieval_bias": "text",
            "sequence": None,
        },
        {
            "id": "C002",
            "question": "Both TP53 and BRCA1 are tumor suppressors involved in DNA damage response. How do their mechanisms differ?",
            "ground_truth": "TP53 (P04637) is a transcription factor that induces expression of cell cycle arrest (CDKN1A) and apoptosis (BAX) genes. BRCA1 (P38398) is an E3 ubiquitin ligase that directly participates in homologous recombination repair of DNA double-strand breaks. TP53 acts mainly at the signaling/transcriptional level; BRCA1 at the repair level.",
            "relevant_uniprot_ids": ["P04637", "P38398"],
            "category": "comparative",
            "difficulty": "hard",
            "retrieval_bias": "text",
            "sequence": None,
        },
        {
            "id": "C003",
            "question": "Compare the roles of glutathione reductase and superoxide dismutase 1 in protecting against reactive oxygen species.",
            "ground_truth": "SOD1 (P00441) converts superoxide (O2•−) to H2O2, the first line of superoxide defense. Glutathione reductase (P00390) regenerates reduced glutathione (GSH) from GSSG using NADPH; GSH is then used by glutathione peroxidase to detoxify H2O2 and lipid peroxides. They operate in complementary steps of the antioxidant network.",
            "relevant_uniprot_ids": ["P00441", "P00390"],
            "category": "comparative",
            "difficulty": "hard",
            "retrieval_bias": "text",
            "sequence": None,
        },
        {
            "id": "C004",
            "question": "How do the iron-handling functions of transferrin and myoglobin differ?",
            "ground_truth": "Serotransferrin (P02787) transports iron (Fe3+) in the blood plasma and delivers it to cells via receptor-mediated endocytosis. Myoglobin (P02144) uses iron within a heme group to store and facilitate intracellular oxygen diffusion in muscle. Transferrin is a transport/storage protein in plasma; myoglobin is an intracellular oxygen-binding protein.",
            "relevant_uniprot_ids": ["P02787", "P02144"],
            "category": "comparative",
            "difficulty": "hard",
            "retrieval_bias": "text",
            "sequence": None,
        },
        {
            "id": "C005",
            "question": "Both TNF and insulin affect metabolic processes. How do their roles in metabolism differ?",
            "ground_truth": "Insulin (P01308) promotes anabolism — stimulating glucose uptake, glycogen synthesis, lipogenesis, and protein synthesis. TNF (P01375) has opposing catabolic and insulin-resistance-promoting effects — it impairs insulin signaling (through IRS-1 serine phosphorylation), promotes lipolysis, and drives the acute-phase response. Elevated TNF contributes to insulin resistance in obesity and type 2 diabetes.",
            "relevant_uniprot_ids": ["P01308", "P01375"],
            "category": "comparative",
            "difficulty": "hard",
            "retrieval_bias": "text",
            "sequence": None,
        },
    ]

    all_questions = text_only + exact_seq + hybrid_q + comparative_q

    out_path = Path("eval/questions_v2.json")
    with open(out_path, "w") as f:
        json.dump(all_questions, f, indent=2)

    # Stats
    cats = {}
    diffs = {}
    biases = {}
    for q in all_questions:
        cats[q["category"]] = cats.get(q["category"], 0) + 1
        diffs[q["difficulty"]] = diffs.get(q["difficulty"], 0) + 1
        b = q["retrieval_bias"]
        biases[b] = biases.get(b, 0) + 1

    print(f"Saved {len(all_questions)} questions to {out_path}")
    print("Categories:", cats)
    print("Difficulties:", diffs)
    print("Retrieval bias:", biases)
    print("No-sequence questions:", sum(1 for q in all_questions if not q["sequence"]))


if __name__ == "__main__":
    main()
