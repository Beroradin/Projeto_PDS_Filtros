# Projeto de Processamento Digital de Sinais e Filtragem de Áudio

Este repositório contém uma aplicação interativa desenvolvida em Python, utilizando o framework Streamlit e a biblioteca SciPy. O objetivo do sistema é fornecer um ambiente prático e visual para o Processamento Digital de Sinais (PDS), com foco especial em sinais de áudio, análise espectral e projeto de filtros digitais.

A arquitetura do projeto foi estruturada para separar a lógica matemática, o gerenciamento de estado e a interface de usuário, garantindo eficiência computacional na manipulação de grandes vetores de dados.

## Funcionalidades Principais

O aplicativo centraliza diversas operações de PDS, organizadas nos seguintes domínios:

* Geração e manipulação de sinais: Criação de ruídos (branco, impulsivo, chirp, alta frequência) e mixagem com controle rigoroso da Relação Sinal-Ruído (SNR).
* Análise espectral: Implementação de Transformada Rápida de Fourier (FFT), Transformada de Fourier de Tempo Curto (STFT) e Transformada Contínua de Wavelet (CWT) para análise no domínio do tempo e da frequência.
* Projeto de Filtros Digitais: Suporte para o dimensionamento de filtros FIR (métodos de janelamento e Parks-McClellan) e IIR (estruturados em seções de segunda ordem para estabilidade numérica).
* Filtragem Adaptativa e Métricas: Aplicação de filtros Least Mean Squares (LMS) para cancelamento de ruído, acompanhados de cálculos de métricas de qualidade como SNR, PSNR e Erro Quadrático Médio (MSE).
* Visualização Otimizada: Algoritmos de downsampling inteligente (Min-Max e LTTB) integrados ao Plotly, permitindo a renderização fluida de sinais longos no navegador sem comprometer o consumo de memória.
