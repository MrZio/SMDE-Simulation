## Metriche

Per ogni nodo (es. Station_1, Station_2, M/M/1):

- c (servers) — numero di server del nodo (parametro c di Kendall)
- N (capacity) — capacità massima del sistema, None se infinita (parametro N di Kendall)
- Lq — lunghezza media della coda di attesa (numero medio di clienti che aspettano, esclusi quelli in servizio)
- L — lunghezza media del sistema (clienti in coda + clienti in servizio)
- rho — utilizzo dei server (frazione di tempo in cui i server sono occupati, 0–1)
- Arrivals — numero totale di clienti arrivati al nodo durante la simulazione
- Served — numero totale di clienti che hanno completato il servizio
- Lost (blocked) — numero di clienti rifiutati perché il sistema era pieno (capacità N raggiunta)

A livello di sistema complessivo (overall):

- Simulation time — tempo simulato totale (orologio finale della simulazione)
- Customers served (system-wide) / n_customers_served — numero di clienti che hanno completato l'intero percorso nel sistema (passando per tutti i nodi)
- Wq (overall, system) — tempo medio di attesa in coda; nota: in un sistema multi-nodo questo valore ha una limitazione (vedi commento nel codice), è rigoroso solo per sistema a singolo nodo
- W (overall, system) — tempo medio totale nel sistema (dall'arrivo iniziale alla partenza finale), corretto anche multi-nodo

Solo per Example 1 (M/M/1 singolo), valori teorici a confronto:

rho, Lq, Wq, L, W teorici — calcolati con le formule chiuse M/M/1 (rho = λ/μ, Lq = ρ²/(1-ρ), ecc.), da confrontare con i valori simulati per la validazione del TRACE