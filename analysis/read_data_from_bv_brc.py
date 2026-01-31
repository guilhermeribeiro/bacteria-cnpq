import pandas as pd

'''
    Como baixar os dados:
    
    1) Acessar o site https://www.bv-brc.org/
    2) Digitar o nome da bactéria que deseja adquirir os dados
    3) Clicar no nome da bactéria na seção "TAXA" do site de busca
    4) Ir na aba "GENOMES" e clicar em download no canto superior esquerdo.
'''

df = pd.read_csv(r'/Users/guilhermeribeiro/MEGA/PyCharmProjects/bacteria-cnpq/data/BVBRC_genome_escherichia_coli.csv')

df_for_evolutionary = df[['Genome Name', 'Strain', 'Assembly Accession', 'Plasmids',
'Chromosome',
'Size',
'GC Content',
'TRNA',
'RRNA',
'CDS',
'Contigs',
'Contig N50',
'Contig L50',
'Contig N50',
'CheckM Completeness',
'CheckM Contamination',
'Genome Quality',
'Genome Status',
'Genome Quality Flags']]

print('')