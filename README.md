# Reconhecimento de Placas com Webcam
Este repositorio expoe uma API simples (`/plates`) que, ao ser acionada, abre a camera do notebook e permanece ativa ate que uma placa seja reconhecida. A janela mostra o video anotado em tempo real e a resposta HTTP devolve a string da placa identificada.

## Fluxo do endpoint `/plates`

1. **Captura via webcam**: inicia `cv2.VideoCapture(0)` e mantem o stream aberto ate que uma placa seja lida.
2. **Deteccao Roboflow**: cada quadro analisado e enviado ao modelo `license-plate-recognition-rxg4e/11` usando o `InferenceHTTPClient`.
3. **Anotacao visual**: `supervision` desenha os bounding boxes e as letras retornadas pelo OCR sobre a janela `Deteccao de Placas`, permitindo acompanhar o que esta sendo lido em tempo real.
4. **OCR com EasyOCR**: cada recorte da placa passa por upsampling, escala de cinza, blur e equalizacao antes da leitura; o texto e limpo (caracteres alfanumericos) e mantido como string.
5. **Resposta HTTP**: o endpoint entrega `{"plate": "ABC1234"}` assim que a string for detectada. Se a camera estiver ocupada, uma resposta HTTP 409 e retornada.

## Requisitos

- Python 3.9+
- Dependencias:
  - `fastapi`
  - `uvicorn`
  - `inference-sdk`
  - `supervision`
  - `opencv-python`
  - `easyocr`
  - `numpy`

Instalacao sugerida:

```bash
pip install fastapi uvicorn inference-sdk supervision opencv-python easyocr numpy
```

> A primeira execucao do EasyOCR faz download automatico dos pesos e pode levar alguns minutos.

## Configuracao

1. Garanta que a camera esteja liberada para aplicativos locais.
2. Ajuste `MODEL_ID` ou `api_key` em `ocr.py` caso utilize outro modelo da Roboflow (a chave padrao continua publica).
3. Opcionalmente, mova a chave para uma variavel de ambiente e leia-a dentro do script.

## Execucao do servidor

Inicie o FastAPI com `uvicorn` (ou simplesmente `python ocr.py`, que ja faz o boot do servidor):

```bash
uvicorn ocr:app --host 0.0.0.0 --port 8000
# ou
python ocr.py
```

## Uso

1. Faca uma requisicao HTTP para `http://localhost:8000/plates` (GET ou POST). Exemplos:
   ```bash
   curl http://localhost:8000/plates
   # ou
   Invoke-RestMethod -Uri http://localhost:8000/plates
   ```
2. A camera sera aberta automaticamente e permanecera ativa exibindo a janela `Deteccao de Placas` ate que alguma placa seja lida.
3. A resposta conter a placa como string unica:
   ```json
   {"plate": "ABC1234"}
   ```

Se a camera estiver em uso por outro processo, o endpoint retorna HTTP 409 (`"A camera ja esta em uso..."`). Erros de hardware retornam HTTP 500.

Adaptado de:  
Plate Extractor: https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/model/11  
Easy OCR: https://github.com/JaidedAI/EasyOCR
