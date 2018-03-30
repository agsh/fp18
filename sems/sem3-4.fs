
(*
set — функция, возвращающая список из
всех атомов, содержащихся в заданном
списке. Каждый атом должен
присутствовать в результирующем списке в
единственном числе.

freq — функция, возвращающая список пар
(символ, частота). Каждая пара определяет
атом из заданного списка и частоту его
вхождения в этот список.
*)




List.zip [1;2;3;4] ['a';'z';'f';'r']



// сами
// zipWith ('a -> 'b -> 'c) -> 'a list -> 'b list -> 'c list

zipWith (+) [1; 2; 3; 4] [4; 3; 2; 1]  // [5, 5, 5, 5]
zipWith (fun x y -> x + " - " + y) ["CSKA"; "Zenith"] ["champion"; "champion"]

let rec zipWith f aList bList =
  match (aList, bList) with
  | [], _ -> []
  | _, [] -> []
  | (x :: xs, y :: ys) -> f x y :: (zipWith f xs ys)

let zip aList bList = zipWith (fun x y -> x, y) aList bList  



let explode (s:string) = [for c in s -> c]
// f:('a -> 'b -> 'c) -> y:'b -> x:'a -> 'c
let flip f y x = f x y

flip List.zip [1;2;3;4;5] (explode "hello") // [('h',1),('e',2),('l',3),('l',4),('o',5)]

let zipReverse a b = flip List.zip a b

// ФВП



// f:('a -> 'b) -> _arg1:'a list -> 'b list
let rec map f = function
  | [] -> []
  | (x::xs) -> f x :: map f xs

map (fun x -> x + 3) [1;5;3;1;6] 
map (List.replicate 3) [3..6]
map (map (fun x -> x ** 2.)) [[1.;2.];[3.;4.;5.;6.];[7.;8.]]
map fst [(1,2);(3,5);(6,3);(2,6);(2,5)]

map fib [0..10]

//f:('a -> bool) -> _arg1:'a list -> 'a list
let rec filter f = function
  | [] -> []
  | (x::xs) when f x -> x :: filter f xs
  | (x::xs) -> filter f xs

filter (fun x -> x > 3) [1;5;3;2;1;6;4;3;2;1]
filter (fun x -> x = 3) [1;2;3;4;5]
filter (fun x -> x <> []) [[1;2;3];[];[3;4;5];[2;2];[];[];[]]
filter (fun x -> List.exists (fun y -> x = y) ['a'..'z']) (explode "u LaUgH aT mE BeCaUsE I aM diFfeRent")
filter (fun x -> List.exists (fun y -> x = y) ['A'..'Z']) (explode "i lauGh At You BecAuse u r aLL the Same")

let sum x y = x + y
let sum = fun x y -> x + y
let sum x = fun y -> x + y

let sumWith9 = sum 9


(*
Сегодня речь пойдет о чрезвычайно общей и постоянно применяющейся в функциональном программировании технике - о свертках и префиксных суммах.
Мы изучим, что они собой представляют и какими бывают, увидим их многочисленные применения и поймем, зачем может быть полезно выразить алгоритм в виде свертки или префиксной сумме.
Рассмотрим вновь несколько рекурсивных функций обработки списков:
*)

let rec sum = function
  [] -> 0
  | h::t -> h + (sum t)

let rec concat = function
  [] -> []
  | h::t -> h @ (concat t)

let rec map f = function
  [] -> []
  | h::t -> (f h)::(map f t)

let rec filter f = function
  [] -> []
  | h::t when (f h) ->  h::(filter f t)
  | h::t -> filter f t

let rec any f = function
  [] -> false
  | h::t -> (f h) || (any f t)

let rec all f = function
  [] -> true
  | h::t -> (f h) && (all f t)

(*Все эти функции вычисляют формулу вида x1 # (x2 # (x3 # (...(xn # u)...)))
Например:
    sum:            +,      0
    min-list:       min,    0 (лучше бы inf)
    concat:         append, []
    map f           f ::,     []
    filter p:       (\x r. if (p x) then (x :: r) else r), []
    any:            (\x r. (p x) || r),         #f
    all:            (\x r. (p x) && r),         #t 
Формулы такого вида над списками называются "правыми свертками". 
А реализовать эту формулу можно так: *)

let rec foldr hash list u =
  match list with
  [] -> u
  | x :: xs -> hash x (foldr hash xs u)

List.foldBack

let sum list = foldr (+) list 0
sum [1..6]

let filter f list = foldr (fun x a -> if f x then x::a else a) list []
filter (fun x -> x%2=0) [1..20]


// Теперь рассмотрим несколько итеративных функций:

let sum =
  let rec loop a = function
    [] -> a
    |h::t -> loop (h+a) t
  loop 0

let concat (list:'a list list) =
  let rec loop a = function
    [] -> a
    |h::t -> loop (a@h) t
  loop [] list

concat [[1..5];[6..7]]

let reverse (list:'a list) =
  let rec loop a = function
    [] -> a
    | h::t -> loop (h::a) t
  loop [] list
reverse [1..9]

(*
Все они вычисляют формулу вида
    (((u # x1) # x2) # .. ) # xn
Например:
    sum:            +,      0
    min:            min,    0 (лучше бы inf)
    dl-concat:      dl-append, dl-empty
    reverse:        ::,       []
    any:            (\r x. r || (p x)),         #f
    all:            (\r x. r && (p x)),         #t
  Формулы такого вида над списками называются "левыми свертками".
  Вот общая функция, вычисляющая левую свертку:
*)

let foldl hash u list =
  let rec loop acc = function
    [] -> acc
    | x :: xs -> loop (hash acc x) xs
  loop u list

let sum = foldl (+) 0

let reverse list = foldl (fun a x -> x::a) [] list
reverse [1..9]

List.fold

(*
Эти две процедуры не просто реализуют один и тот же алгоритм ("свертку") двумя разными способами - это два разных алгоритма, предназначенных для разных целей.
Первый из них (левая свертка) - это итеративный алгоритм, начинающий с некоторого начального значения и модифицирующий его с помощью каждого элемента списка. В императивном языке ему соответствует цикл:
    Result res = u;
    for(Value v in list) {
        res #= v;
    }
    return res;
Именно этот распространенный паттерн абстрагируется левой сверткой.
Он основан на том, что известно правило изменения ответа при дописывании элемента в конец списка.
Например:
    При дописывании элемента в конец списка его сумма увеличивается на этот элемент
    При дописывании элемента в конец списка к его reverse приписывается этот элемент
Итеративность левой свертки основана на том, что можно итеративно представить список в виде последовательности дописываний элемента в конец, начиная с пустого списка.
*)

// Хвостовая рекурсия
let rec fac' = function
  0 -> 1
  | x -> x * fac' (x-1)
  
let rec fac'' x =
  if x <= 0I then 1I
  else x * fac'' (x-1I)

fac'' 1000000I

let fac x =
  let rec fac x acc = 
    if x <= 0I then acc
    else fac (x - 1I) (acc * x)
  fac x 1I

fac 1000000I

(* Генерация списков *)
let list1 = [1 .. 10]       // val it : int list = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]
let list2 = [1 .. 2 .. 10]  // val it : int list = [1; 3; 5; 7; 9]
let list3 = ['a' .. 'g']    // val it : char list ['a'; 'b'; 'c'; 'd'; 'e'; 'f'; 'g';]

// Ключевая конструкция yield или -> для генерации списков
let list4 = [ for a in 1 .. 10 do yield (a * a) ] // val it : int list = [1; 4; 9; 16; 25; 36; 49; 64; 81; 100]

// Несколько элементов
let list5 = 
  [ for a in 1 .. 3 do
      for b in 3 .. 7 do
        yield (a, b) ]
// val it : (int * int) list = [(1, 3); (1, 4); (1, 5); (1, 6); (1, 7); (2, 3); (2, 4); (2, 5); (2, 6); (2, 7); (3, 3); (3, 4); (3, 5); (3, 6); (3, 7)]

// Синтаксический сахар для замены do-yield: ->
let list6 = 
  [for a in 1..3 do
    for b in 4..6 ->
      (a, b) ]

// генерация списка с условием
let list7 = [ for a in 1 .. 100 do 
                if a % 3 = 0 && a % 5 = 1 then yield a]
// val it : int list = [6; 21; 36; 51; 66; 81; 96]

// для любых перечислимых типов
let list8 = [for a in ['a'.. 'f'] do yield [a; a; a] ]
// val it : char list list = [['a'; 'a'; 'a']; ['b'; 'b'; 'b']; ['c'; 'c'; 'c']; ['d'; 'd'; 'd']; ['e'; 'e'; 'e']; ['f'; 'f'; 'f']]

// yield! используется для генерации одновременно нескольких элементов
let list9 = [for a in 1 .. 5 do yield! [ a .. a + 3 ] ] // val it : int list = [1; 2; 3; 4; 2; 3; 4; 5; 3; 4; 5; 6; 4; 5; 6; 7; 5; 6; 7; 8]

// для генерации списка можно использовать различные возможности языка
let list10 = 
  [
    let thisIsIt a = a + "?"
    for a in 1 .. 5 do
      match a with
      | 3 -> yield! ["hello"; "world"; thisIsIt "!"]
      | _ -> yield a.ToString()
  ] |> List.fold (fun x y -> x + " " + y) ""
// val it : string list = ["1"; "2"; "hello"; "world"; "!"; "4"; "5"]


// последовательности задаются похожим на списки образом
let seq1 = seq {1..78}
seq1 |> Seq.iter (printfn "%d")
// если нужно задать последовательность перечислением элементов, используются квадратные скобки
let seq2 = seq [1; 2; 9]
List.toSeq [1; 2; 9]
// последовательность может быть сколь угодно большой, т.к. элементы последовательности высчитываются только тогда, когда они требуются
let seq3 = seq { 1I .. 1000000000000I }
let list11 = [1I .. 1000000000000I ]

// ещё один пример
let seq4 =
    seq { for a in 1 .. 10 do
            printfn "created: %i" a
            yield a }
// val seq4 : seq<int>, 
seq4 |> Seq.take 3
(*
а вот когда нам нужен 3-ий элемент последовательности, начинаются вычисления
created: 1
created: 2
created: 3
val it : seq<int> = seq [1; 2; 3]
*)


// ещё один пример сравнения последовательностей и списков
let list12 = [for a in 5..-1..0 -> 10 / a] // ошибка деления на ноль
let seq5 = seq {for a in 5..-1..0 -> 10 / a} // ошибок нет
seq5 |> Seq.take 5 |> Seq.iter (printfn "%i") // ошибок нет
seq5 |> Seq.take 6 |> Seq.iter (printfn "%i") // 2, 2, 3, 5, 10, ошибка деления на ноль

// бесконечные последовательности
// чётные числа
let seq6 =
  let rec seq6' x = seq { yield x; yield! seq6' (x + 2) } // рекурсивное создание последовательности
  seq6' 0

Seq.take 10 seq6 // val it : seq<int> = seq [0; 2; 4; 6; ...]
seq6 |> Seq.iter (printfn "%d")

// пример из самостоятельной работы
let seq7 = 
    let rec seq7' x = seq { yield! [0; x]; yield! seq7' (x + 1) }
    seq7' 1;;
Seq.take 10 seq7 // val it : seq<int> = seq [0; 1; 0; 2; ...]

let rec ones' = 1 :: ones' // бесконечный список из единиц как в Haskell'е не сделать..
let rec ones = seq {yield 1; yield! ones} // но его можно реализовать так
Seq.take 100 ones // val it : seq<int> = seq [1; 1; 1; 1; ...]
let ones'' = Seq.initInfinite (fun _ -> 1) // или так, с использованием функции initInfinite

(* 
Генерация последовательностей с использованием функции Seq.unfold
тип этой функции ('a -> ('b * 'a) option) -> 'a -> seq<'b>
первый аргумент - функция для генерации нового элемента последовательности из состояния, она должна возвращать известный нам тип option, заключающий в себе кортеж из этого самого нового элемента и нового состояния
второй аргумент - начальное состояние
*)
// создаём список из чётных чисел меньше ста
let seq8 = Seq.unfold (fun state -> if state <= 100 then Some(state, state+2) else None) 2
seq8 |> Seq.iter (printf "%i ")

// генерация бесконечной последовательности чисел Фиббоначе
let fibb = Seq.unfold (fun state -> Some(fst state + snd state, (snd state, fst state + snd state))) (0,1)
Seq.take 20 fibb
Seq.iter (printf "%i ") fibb
