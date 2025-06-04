# Zasady gry Literaki

## 1. Plansza
Gra odbywa sie na planszy 15x15. W kodzie Python kazde pole jest opisane slownikiem zawierajacym m.in. mnoznik slowa (`word_mult`), kolor pola (`board_color`) i informacje, czy jest to pole startowe (`is_start`).

## 2. Rozpoczecie gry
- Gracze dobieraja po 7 płytek i losowo ustala sie osobe rozpoczynajaca.
- Pierwsze slowo musi liczyc co najmniej 2 litery i przechodzic przez centralne pole startowe (daje podwojna premie slowa).

## 3. Tura gracza
W trakcie swojej kolejki gracz moze:
1. **Polozyc plytki** w jednym rzędzie lub kolumnie, tworzac poprawne slowa krzyzujace sie z istniejacymi na planszy.
2. **Wymienic plytki** (maksymalnie 3 razy w trakcie calej gry, jesli w puli pozostaje >=7 płytek), rezygnujac przy tym z ruchu.
3. **Pominac kolejke.**

## 4. Punktacja
- Kazda litera ma wartosc punktowa zalezną od jej koloru/kategorii (1, 2, 3 lub 5 punktow).
- Jesli kolor plytki odpowiada kolorowi pola, wartosc litery w tym ruchu jest potrojona.
- Pola `2x` i `3x` zwiekszaja wartosc calego slowa odpowiednio dwukrotnie i trojkotnie; mnozniki moga sie kumulowac.
- Zuzycie wszystkich 7 płytek w jednym ruchu nagradzane jest bonusem 50 punktow.

## 5. Zakoncznie gry
- Gra konczy sie, gdy zabraknie płytek w puli i jeden z graczy wykorzysta wszystkie swoje płytki albo gdy wszyscy gracze dwukrotnie z rzędu pominą ruch.
- Po zakończeniu od wyniku każdego gracza odejmuje się wartość niewykorzystanych płytek. Gracz, ktory zakonczyl gre wylozeniem wszystkich swoich płytek, zyskuje sumy odjetych punktow przeciwnikow.
- Wygrywa gracz z najwyzszym wynikiem (mozliwy remis).

## 6. Blanki
- W grze wystepuja dwa blanki (płytki o wartosci 0). Po polozeniu blanku gracz deklaruje, jaka litere reprezentuje. Blanki nie korzystaja z premii za kolor.
