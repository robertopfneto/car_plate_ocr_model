# Reconhecimento de Placas com OCR
Este repositório contém um fluxo simples para detectar placas veiculares em uma imagem e executar OCR sobre os recortes gerados. O script principal, `ocr.py`, integra o serviço **Roboflow Inference**, a biblioteca de anotações **Supervision** e o **EasyOCR** para entregar uma solução ponta a ponta.

## Fluxo do `ocr.py`

1. **Detecção de placas**: envia a imagem `placabrasil.jpg` a um modelo hospedado na Roboflow (`license-plate-recognition-rxg4e/11`) por meio do `InferenceHTTPClient`.
2. **Anotação**: reconstrói as detecções com `supervision`, desenha bounding boxes sobre a imagem original e exibe o resultado.
3. **Recorte das placas**: cria a pasta `img_placas/` e salva cada placa individual encontrada com a nomenclatura `placa_<n>.jpg`.
4. **Pré-processamento para OCR**: amplia cada recorte, converte para tons de cinza, aplica blur gaussiano e equalização de histograma para melhorar o contraste.
5. **Leitura com EasyOCR**: identifica textos em cada placa e grava as versões anotadas em `img_placas/ocr_<arquivo>.jpg`, além de imprimir o texto reconhecido no terminal.

## Requisitos

- Python 3.9+ (recomendado)
- Dependências:
  - `inference-sdk`
  - `supervision`
  - `opencv-python`
  - `easyocr`
  - `numpy`

Instalação sugerida:

```bash
pip install inference-sdk supervision opencv-python easyocr numpy
```

> 💡 A primeira execução do EasyOCR fará o download automático dos pesos necessários — tenha paciência, pois pode demorar alguns minutos.

## Configuração

1. Coloque a imagem `placabrasil.jpg` (ou uma imagem com placas de interesse) na raiz do projeto.
2. Ajuste o `model_id` em `ocr.py` caso utilize outro modelo da Roboflow.
3. (Opcional) Substitua a `api_key` diretamente no script por uma variável de ambiente para evitar expor credenciais em repositórios.

Exemplo usando variável de ambiente no PowerShell:


```powershell
$Env:ROBOFLOW_API_KEY="wsNv1Yo4iSAU0CuXa0mI"
python ocr.py
```

Não tem problema deixar essa chave de api assim pois ela é do próprio código disponibilizado no projeto roboflow (https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/model/11)

## Execução

```bash
python ocr.py
```

Ao finalizar, o script exibirá a imagem anotada e registrará os textos lidos de cada placa no console.

## Estrutura de Saída

- `img_placas/placa_<n>.jpg`: recorte cru de cada placa detectada.
- `img_placas/ocr_<arquivo>.jpg`: recorte com o texto reconhecido desenhado sobre a imagem.
- Mensagens no terminal com o texto detectado para cada recorte.


Adaptado de:
Plate Extractor: https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/model/11
Easy OCR: https://github.com/JaidedAI/EasyOCR
