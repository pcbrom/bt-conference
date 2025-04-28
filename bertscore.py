# Instalar o pacote se necessário
# pip install bert-score

from bert_score import score

# Textos original e parafraseado (em português)
q_text = ("Um pintor pretende fazer uma reprodução do quadro Guernica em uma tela de dimensões 20 cm por 30 cm. "
          "Essa obra, de autoria do espanhol Pablo Picasso, é uma pintura com 3,6 m de altura e 7,8 m de comprimento. "
          "A reprodução a ser feita deverá preencher a maior área possível da tela, mantendo a proporção entre as dimensões da obra original. "
          "A escala que deve ser empregada para essa reprodução é:")

hat_q_text = ("Um artista quer criar uma réplica da pintura Guernica em uma tela de 20 cm por 30 cm. "
              "O original, feito por Pablo Picasso, mede 3,6 metros de altura por 7,8 metros de largura. "
              "A réplica deve ocupar a maior área possível da tela sem alterar a proporção da obra. "
              "Qual deve ser a escala usada?")

# Calculando o BERTScore
P, R, F1 = score([hat_q_text], [q_text], lang="pt", rescale_with_baseline=True)

# Exibindo o F1-score
print(f"BERTScore (F1) = {F1.item():.4f}")
