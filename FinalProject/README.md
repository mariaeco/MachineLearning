# 📚 PREDIÇÃO DE DESEMPENHO DE ALUNOS DA PARAÍBA NO ENEM

### 🚺  Autores: Maria Cardoso, Indrid Ferreira & Paloma Duarte



### 📝 Descrição
Este projeto aplica técnicas de aprendizado de máquina para analisar e prever o desempenho no Enem de estudantes com base em 1998 a 2023.



**INTRODUÇÃO**
 
 Os dados de desempenho no Enem são uma importante ferramenta para avaliação por gestores escolares a nível Federal, Estadual, Municipal e Privado, para tomadas de decisões. Entretanto, os dados fornecidos pelo INEP, embora disponíveis facilmente para baixar no [site](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados) são de difícil manipulação, e apenas profissionais com algum conhecimento de programação conseguem de fato acessar, abrir e explorar os dados. 
 


**OBJETIVO GERAL:**

   Aplicar técnicas de aprendizado de máquina para predição de aspectos que auxiliem no desempenho de estudantes no Enem, identificando padrões que podem ajudar a melhorar estratégias educacionais e oferecer suporte personalizado aos alunos.

   

**Objetivos Específicos:**
  
  
   - Descrever o perfil dos inscritos no Enem (Sexo, Raça, Tipo de Escola, e aspectos socioecômicos, como renda, número de pessoas na casa, bens tecnológicos).
   - Descrever o perfil dos faltantes
   - Descrever as médias gerais e por componentes curriculares (Linguagens, Ciências Humanas, Ciências da Natureza, Matemática e Redação).
   - Relacionar as Médias as variaveis de perfil sócio-econômico.
   - Tentar predizer os asp56ectos mais importantes para a melhoria das notas.
 
 

 🛠️ Tecnologias e Bibliotecas

        Python
        Pandas
        NumPy
        Scikit-Learn
        Matplotlib
        Seaborn


**Variáveis Utilizadas**

### 📂 Estrutura de Arquivos do Projeto

- **Projeto_Final_Educ.ipynb**: Contém scripts e notebooks relacionados à análise e visualização dos dados educacionais, incluindo gráficos e relatórios descritivos.

- **Projeto_Final_Enem.ipynb**: Inclui os modelos de aprendizado de máquina desenvolvidos para predição de desempenho no Enem, bem como os experimentos realizados com diferentes algoritmos.

- **modelos.py**: módulo com os modelos DecisionTree, SVM e Redes Neurais

- **Tratamentos de Dados**:
   - **includingLabels.ipynb**: Script responsável por adicionar rótulos e categorias aos dados brutos, facilitando a análise e modelagem.
   - **dataPB_Selection.ipynb**: Realiza a seleção de dados específicos da Paraíba, filtrando informações relevantes para o estudo.
   - **data_cleaning.ipynb**: Contém funções para limpeza e pré-processamento dos dados, como remoção de valores ausentes, normalização e transformação de variáveis.

- **Analise Descritiva**:
   - **AnaliseDescritiva.ipynb**: scripts com análise geral para entendimento do dado.