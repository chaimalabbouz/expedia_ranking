le traitement que on va le faire :
1) enlever la colone position par ce que elle n existe pendant la predection (ces l ordre d affichage par exepdia )
2) de meme pour"gross_bookings_usde , c est le montant total de la réservation,une fois que la resarvation est faite, cest une fuite de données .
3) "random_bool" vaut 1 si les hôtels ont été affichés dans un ordre aléatoire, et 0 si l’ordre était celui de l’algorithme d’Expedia. c est pourquoi on doit enlever 

* les 2 labels:clink_bool + booking_bool
