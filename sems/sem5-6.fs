
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
open System
open System.Windows.Forms



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
Seq.take 20 fibb |> Seq.iter (printf "%i ")
Seq.iter (printf "%i ") fibb

Список треугольных чисел
Список пирамидальных чисел

n-е треугольное число tn равно количеству
одинаковых монет, из которых можно построить
равносторонний треугольник, на каждой стороне
которого укладывается n монет.

n-е пирамидальное число pn равно количеству
одинаковых шаров, из которых можно построить
пирамиду с треугольным основанием, на каждой
стороне которой укладывается n шаров


List.fold (+) 0 [1..10] 
List.scan (+) 0 [1..10]
List.fold (+) 0 [10..-1..1]
List.scan (+) 0 [10..-1..1]



let tr = Seq.unfold (fun (sum, n) -> Some(sum, (sum + n, n + 1))) (0, 1) 

let tr = 
  let rec tr' sum n = seq {yield sum; yield! tr' (sum + n) (n + 1)}
  in tr' 0 1

let pir = seq { for a in tr do
    yield a
  } 

let tr =
  let rec tr' n = 
    match n with
    | n when n <= 1 -> 1
    | n -> n + tr' (n-1)
  in Seq.map (tr') (Seq.initInfinite (fun x -> x + 1))  

let tr = Seq.scan (+) 1 (Seq.initInfinite (fun x -> x + 2))
let pir = Seq.scan (+) 0 tr |> Seq.tail  

Seq.iter (printf "%i ") tr 

(*Правая же свертка - рекурсивный алгоритм, строящий составное значение из головы списка и ответа для хвоста списка.
В данном случае известно правило изменения ответа при приписывании элемента в начало списка.
Например:
    При приписывании элемента в начало списка его сумма увеличивается на этот элемент
    При приписывании элемента в начало списка списков к его конкатенации приписывается этот элемент
В общем случае простого и эффективного способа преобразования между этими двумя правилами не существует (нельзя легко понять, как изменяется результат от дописывания в конец, зная, как изменяется результат от дописывания в начало, и наоборот).*)

foldl (+) 0 [1..5]
foldl (-) 0 [1..5] // -15 = 0-1-2-3-4-5
foldr (-) 0 [1..5] // 3 = 1-(2-(3-(4-(5-0))))

List.fold (+) 0 [1..5]
List.foldBack (+) [1..5] 0

List.fold (-) 0 [1..5]
List.foldBack (-) [1..5] 0


(*
Однако в некоторых случаях результаты левой и правой свертки с одной и той же операцией и начальным значением - совпадают. В каких же?
    (((u # x1) # x2) # .. ) # xn
    x1 # (x2 # (x3 # (... # u)))
Записав эти формулы для списков из 1 или 3 элементов, получаем:
    forall x, u#x = x#u  - следовательно, во-первых, операция # должна оперировать над аргументами одинакового типа, а во-вторых, u должно коммутировать с каждым элементом этого типа.
    Отсюда НЕ следует, что u должно быть единицей для # - однако очень часто выбирают u именно таким образом. Рассмотрим именно этот случай.
    forall a b c, ((u#a)#b)#c = a#(b#(c#u)) --> (a#b)#c = a#(b#c) - т.е. операция # должна быть ассоциативна.
  Эти два условия достаточны (но, если u не единичный элемент, не необходимы) для того, чтобы результаты левой и правой свертки совпадали.
(вскоре мы увидим, что если функция выражается с *разной* операцией #, но с одинаковым значением u для левой и правой свертки, то имеет место гораздо более сильное свойство - Третья теорема о гомоморфизмах.)
 
 Помимо сверток над списками часто используются также "бегущие свертки" (scan), называемые "префиксными суммами": например, "бегущая сумма", "бегущий минимум". Их можно интерпретировать как вычсления последовательности промежуточных результатов свертки.
   *)

scanl, scanr

List.scanBack (+) [1..5] 0
List.scan (+) 0 [1..5]





// coins

// fibs = 0 : 1 : zipWith (+) fibs (tail fibs)

let zipWith f seq1 seq2 = Seq.zip seq1 seq2 |> Seq.map (fun (x, y) -> f x y)
let rec fibs = seq {yield 0; yield 1; yield! (zipWith (+) fibs (Seq.tail fibs))}

Seq.take 20 fibs |> Seq.iter (printf "%A ")

(*
type Anniversary =
 Birthday of string * int * int * int
 | Wedding of string * string * int * int * int
 | Death of string * int * int * int
 
Birthday ("someone", 2000, 5, 4)

let today = Birthday ("someone", 2000, 5, 4)
*)

type Year = int
type Month = int
type Day = int
type Date = Day * Month * Year
type Name = string
type Where = string
type Anniversary =
  | Birthday of Name * Date
  | Wedding of Name * Name * Date
  | Death of Name * Date * Where

Birthday ("Someone", (31, 3, 2018))

let (kurtCobain : Anniversary) = Birthday ("Kurt Cobain", (1967, 2, 20))
let (kurtWedding : Anniversary) = Wedding ("Kurt Cobain", "Courtney Love", (1990, 1 ,12))
let anniversaries = [
    kurtCobain;
    kurtWedding;
    Death ("Kurt Cobain", (1994, 4, 5), "At home")
]

let showDate d m y = d.ToString() + "." + m.ToString() + "." + y.ToString()

let showAnniversary = function
  Birthday (name, (year, month, day)) -> name + " born " + showDate year month day // синеньким
  | Wedding (name1, name2, (year, month, day)) ->
      name1 + " married " + name2 + " on " + showDate year month day
  | Death (name, (year, month, day), where) -> name + " dead in " + showDate year month day

let who = function
 Birthday (him, _) -> him 
 | Wedding (him, _, _) -> him
 | Death (him, _, _) -> him

List.map who anniversaries

Seq.zip (Seq.initInfinite (fun x -> x + 1)) anniversaries 
  |> Seq.map (fun (num, ann) -> num.ToString() + "). " + showAnniversary ann)
  |> Seq.iter (printf "%A\n") 

(*

1) Kurt Cobain born 1967-2-20
2) Kurt Cobain married Courtney Love on 1990-1-12
3) Kurt Cobain dead 1994-4-5

*)

Seq.zip (Seq.initInfinite (fun x -> x + 1)) anniversaries
  |> Seq.map (fun (num, ann) -> num.ToString() + ") " + showAnniversary ann)
  |> fun list -> Seq.foldBack (fun x y -> x + "\n" + y) list ""

type Point = { x : float; y : float }

let a = { x = 13.22 ; y = 8.99 }
let b = { a with y = 666.13 }
let absPoint a = sqrt (a.x*a.x + a.y*a.y)


Some 1
None
Some "str" // type?
Some 42
Some 42 :: [None]  // ?
Some 42 :: [Some "str"; None] // ?

type Option<'a> =
  Some of 'a
  | None

let getExactValue opt =
  match opt with
  None -> failwith "error"
  | Some v -> v   

type 'a Option =
  Some of 'a
  | None

Some 5 
None 

type 'a List =  // haskell!!!
  Nil
  | (Cons) of 'a * ('a List)

let l1 = Cons (3, (Cons (4, (Cons (5, Nil)))))
let l2 = Cons (2, l1)

let rec apply x y = 
  match x with
    | Nil -> y
    | Cons (head, tail) -> Cons (head, apply tail y)

apply l1 l2


type SetType = int -> bool


let (a:SetType) = (fun a -> true)

let contains (s:SetType) (a:int) = s a

contains a 100

let singletonSet b : SetType = fun a -> a = b

let b = singletonSet 3

let doubleSet a b = fun c -> (a = c) || (b = c)

let c = doubleSet 1 3
contains c 6

let singletonSet (b:int) : SetType = fun (a:int) -> a = b
let singletonSet b : SetType= fun (a:int) -> a = b
let singletonSet (b:int) : SetType = fun a -> a = b

let singletonSet : SetType = fun b -> (fun (a: int) -> (a = b))

let singletonSet b =
  let answer a = a=b
  (answer:Set)

let d = singletonSet 5
let e = singletonSet 6
let g = singletonSet 7
contains d 5
contains d 6

let union setA setB : Set =
  let set = fun a -> (contains setA a) || (contains setB a)  
  (set:Set)

let f = union d e
contains f 7

// union, intersect, diff, filter, forAll, exists 

let intersect (setA:Set) (setB:Set) : Set =

let diff (setA:Set) (setB:Set) : Set =

let filter (set:Set) (f: 'a -> bool) : Set =

let range = [-1000..1..1000]

let forAll (set:Set) (f: 'a -> bool) : bool =

let exists (set:Set) (f: 'a -> bool) : bool =






let intersect setA setB : Set =
  fun a -> (contains setA a) && (contains setB a)  
let diff setA setB : Set =
  fun a -> (contains setA a) && not (contains setB a)  
let filter = intersect
let filter setA f : Set =
  fun a -> (contains setA a) && (f a)  


let range = [-1000..1..1000]
let forAll set f : bool = List.fold (fun acc x -> (not (contains set x) || f x) && acc) true range 
let exists set f : bool = not (forAll set (not f))




type Tree<'a> =
  EmptyTree
  | Node of 'a * 'a Tree * 'a Tree

let singleton x = Node (x, EmptyTree, EmptyTree)

singleton 'c'

singleton (singletonSet 6)

singleton (fun x -> x - 1)

let rec treeInsert x tree = match tree with
  EmptyTree -> singleton x 
  | Node (a, left, right) -> 
    if x = a then Node (x, left, right) 
    else 
      if x < a then Node (a, (treeInsert x left), right) 
      else Node (a, left, (treeInsert x right))
// when 'a : comparsion

let list2tree list =
 let rec l2t acc list = match list with
   [] -> acc
   | (head::tail) -> l2t (treeInsert head acc) tail
 in l2t EmptyTree list

let t = list2tree [15; 6; 3; 5; 13; 98; 54; 12; 1; 6; 4; 90; 9] 


 // list2tree через fold



let flip f x y = f y x
let list2tree list = List.fold (flip treeInsert) EmptyTree list
let list2tree list = List.foldBack treeInsert list EmptyTree

list2tree [15; 6; 3; 5; 13; 98; 54; 12; 1; 6; 4; 90; 9] 



let rec tree2list tree = 
  match tree with
  EmptyTree -> []
  | Node (a, left, right) -> (tree2list left) @ (a :: (tree2list right))

tree2list // сами


let treesort x = x |> list2tree |> tree2list

treesort [12; 1; 6; 4; 90; 9]

list2tree [12; 12; 12; 13; 13; 14]
Как будет выглядеть дерево?

type 'a Tree =
  EmptyTree
  | Node of 'a * int * 'a Tree * 'a Tree

let singleton x = ? 

let rec treeInsert x = ?


let rec foldTree treeFunction listValue tree =
  match tree with
  EmptyTree -> listValue
  | Node (a, left, right) -> treeFunction a (foldTree treeFunction listValue left) (foldTree treeFunction listValue right)


let foldTree treeFunction listValue tree =
    let rec loop tree cont =
        match tree with
        | EmptyTree -> cont listValue
        | Node (x, left, right) -> loop left (fun leftAcc -> 
            loop right (fun rightAcc -> 
              cont (treeFunction x leftAcc rightAcc)
            )
          )
    loop tree (fun x -> x)
    
// написать foldTree без продолжений


let sumTree tree = foldTree (fun (x:int) left right -> x + left + right) 0 tree
[2;7;4;3;5;8] |> list2tree |> sumTree
let a = [2;7;4;3;5;8] |> list2tree
sumTree a

let heightTree tree = foldTree (fun _ left right -> 1 + max left right) 0 tree
[2;7;4;3;5;8] |> list2tree |> heightTree 

let tree2List tree = foldTree (fun x left right -> left @ (x :: right)) [] tree
[2;7;4;3;5;8] |> list2tree |> tree2List

// найти максимальное значение в дереве

// перевернуть дерево



open System

let generate min max = 
  let rnd = Random()
  rnd.Next(min, max)

generate 0 42

let rnd = Random()
// сгенерировать дерево
let rec generateTree () =
  match (rnd.Next(0, 42)) with
  | n when n < 20 -> Node (n, generateTree (), generateTree ())
  | _ -> EmptyTree

generateTree ()

// проверить, что два дерева подобны

let rec exactTree tree1 tree2 =
  match (tree1, tree2) with
  | (EmptyTree, EmptyTree) -> true
  | (EmptyTree, Node _) -> false
  | (Node _, EmptyTree) -> false
  | (Node (_, l1, r1), Node (_, l2, r2)) -> (exactTree l1 l2) && (exactTree r1 r2)

type 'a Tree =
  | EmptyTree
  | Leaf of 'a
  | Node of 'a * 'a Tree list

Node(1, [Leaf 3; EmptyTree; Node(2, [])])

let rec get_depth tree =
  let rec dep' tree =
    match tree with
    | EmptyTree -> 0
    | Leaf x -> 1
    | Node(x, list) -> List.max (List.map dep' list)
  dep' tree  

// найти высоту дерева, EmptyTree не считаются

type Tree<'LeafData,'INodeData> =
    | LeafNode of 'LeafData
    | InternalNode of 'INodeData * Tree<'LeafData,'INodeData> seq

let tree = InternalNode (2, seq [ InternalNode (3, seq [LeafNode ('n')]); LeafNode ('a') ])    

let rec fold fLeaf fNode acc (tree:Tree<'LeafData,'INodeData>) : 'r = 
    let recurse = fold fLeaf fNode  
    match tree with
    | LeafNode leafInfo -> 
        fLeaf acc leafInfo 
    | InternalNode (nodeInfo, subtrees) -> 
        Seq.fold recurse (fNode acc nodeInfo) subtrees 

fold (fun acc lD -> acc + 1) (fun acc nD -> acc + 1) 0 tree

let rec map fLeaf fNode (tree:Tree<'LeafData,'INodeData>) = ?

type FileInfo = {name:string; fileSize:int}
type DirectoryInfo = {name:string; dirSize:int}

type FileSystemItem = Tree<FileInfo,DirectoryInfo>

let fromFile (fileInfo:FileInfo) = 
    LeafNode fileInfo 

let fromDir (dirInfo:DirectoryInfo) subitems = 
    InternalNode (dirInfo,subitems)

let readme : Tree<FileInfo,DirectoryInfo> = fromFile {name="readme.txt"; fileSize=1}
let config = fromFile {name="config.json"; fileSize=2}
let build  = fromFile {name="build.sh"; fileSize=3}
let src = fromDir {name="src"; dirSize=10} [readme; config; build]
let bin = fromDir {name="bin"; dirSize=10} []
let root = fromDir {name="root"; dirSize=5} [src; bin]

let totalSize fileSystemItem =
    let fFile acc (file:FileInfo) = 
        acc + file.fileSize
    let fDir acc (dir:DirectoryInfo)= 
        acc + dir.dirSize
    fold fFile fDir 0 fileSystemItem 

readme |> totalSize  
src |> totalSize     
root |> totalSize    

// largestFile : fileSystemItem:Tree<FileInfo,'a> -> FileInfo option
let largestFile fileSystemItem =




// to remove
let largestFile fileSystemItem =
    let fFile (largestSoFarOpt:FileInfo option) (file:FileInfo) = 
        match largestSoFarOpt with
        | None -> 
            Some file                
        | Some largestSoFar -> 
            if largestSoFar.fileSize > file.fileSize then
                Some largestSoFar
            else
                Some file
    let fDir largestSoFarOpt dirInfo = 
        largestSoFarOpt
    fold fFile fDir None fileSystemItem

readme |> largestFile
src |> largestFile
bin |> largestFile
root |> largestFile

let rec map fLeaf fNode (tree:Tree<'LeafData,'INodeData>) = ?


let rec map fLeaf fNode (tree:Tree<'LeafData,'INodeData>) = 
  let recurse = map fLeaf fNode  
  match tree with
  | LeafNode leafInfo -> 
    let newLeafInfo = fLeaf leafInfo
    LeafNode newLeafInfo 
  | InternalNode (nodeInfo,subtrees) -> 
    let newSubtrees = subtrees |> Seq.map recurse 
    let newNodeInfo = fNode nodeInfo
    InternalNode (newNodeInfo, newSubtrees)




open System
open System.IO

DirectoryInfo("/home/und/fsharp")

type FileSystemTree = Tree<FileInfo,DirectoryInfo>

let fromFile (fileInfo:FileInfo) = 
    LeafNode fileInfo 

let rec fromDir (dirInfo:DirectoryInfo) = 
    let subItems = seq {
        yield! dirInfo.EnumerateFiles() |> Seq.map fromFile
        yield! dirInfo.EnumerateDirectories() |> Seq.map fromDir
    }
    InternalNode (dirInfo,subItems)

let totalSize fileSystemItem =
    let fFile acc (file:FileInfo) = 
        acc + file.Length
    let fDir acc (dir:DirectoryInfo)= 
        acc 
    fold fFile fDir 0L fileSystemItem 
   
let currentDir = fromDir (DirectoryInfo("/home/und/fsharp"))

let torDir = fromDir (DirectoryInfo("/home/und/tor-browser_en-US"))

currentDir |> totalSize  
torDir |> totalSize

let largestFile fileSystemItem:Tree<FileInfo,'a> : FileInfo option = ?






// to remove
let largestFile fileSystemItem =
    let fFile (largestSoFarOpt:FileInfo option) (file:FileInfo) = 
        match largestSoFarOpt with
        | None -> 
            Some file                
        | Some largestSoFar -> 
            if largestSoFar.Length > file.Length then
                Some largestSoFar
            else
                Some file

    let fDir largestSoFarOpt dirInfo = 
        largestSoFarOpt

    fold fFile fDir None fileSystemItem

currentDir |> largestFile  
torDir |> largestFile

let dirListing fileSystemItem =
    let printDate (d:DateTime) = d.ToString()
    let mapFile (fi:FileInfo) = 
        sprintf "%10i  %s  %-s"  fi.Length (printDate fi.LastWriteTime) fi.Name
    let mapDir (di:DirectoryInfo) = 
        di.FullName 
    map mapFile mapDir fileSystemItem

currentDir 
    |> dirListing 

