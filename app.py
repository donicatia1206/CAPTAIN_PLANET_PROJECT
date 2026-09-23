from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)
CORS(app)

# Matriz de Treino da Árvore de Decisão: [BPM, Estresse]
X = [
    [70, 15],   # Gratidão
    [45, 20],   # Paz
    [110, 30],  # Alegria
    [85, 60],   # Tristeza
    [95, 85],   # Dor da Alma
    [130, 90]   # Raiva
]

y = ["GRATIDAO", "PAZ", "ALEGRIA", "TRISTEZA", "DOR_DA_ALMA", "RAIVA"]

# Treino do modelo
clf = DecisionTreeClassifier()
clf.fit(X, y)

mensagens = {
    "GRATIDAO": "Sinta a beleza de estar vivo e presente. 🌻",
    "PAZ": "Equilíbrio interior e quietude na alma. 🕊️",
    "ALEGRIA": "Energia radiante que renova todo o seu ser! ✨",
    "TRISTEZA": "A lágrima limpa a visão para o próximo recomeço. 🌧️",
    "DOR_DA_ALMA": "Dor da alma é aprendizado e força em transformação. 💙",
    "RAIVA": "Força bruta que precisa ser canalizada com sabedoria. 🔥"
}

@app.route('/analisar', methods=['POST'])
def analisar_biometria():
    dados = request.get_json()
    bpm = float(dados.get('bpm', 70))
    estresse = float(dados.get('estresse', 0))
    
    predicao = clf.predict([[bpm, estresse]])[0]
    resposta_poetica = mensagens.get(predicao, "Estado em sintonização...")
    
    return jsonify({
        "estado": predicao,
        "mensagem": resposta_poetica
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)