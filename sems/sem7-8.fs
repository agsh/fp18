open System
open System.Net
open System.Threading

/////////////////////////////////////////
// Computation Expressions

[<AbstractClass>]
type M<'a> =
    abstract Bind : M<'a> -> ('a -> M<'b>) -> M<'b>
    abstract Return : 'a -> M<'a>
    abstract ReturnFrom : M<'a> -> M<'a>

///////////////////

let oReturn (x: 'a) : 'a option = Some(x)

let oReturnFrom (x: 'a option) : 'a option = x

let oBind (mx: 'a option) (f: 'a -> 'b option) : 'b option = 
    match mx with
    | Some(x) -> f x
    | None -> None
                                                             
let (>>=) = oBind

type OptionMonad() =
    member x.Bind(p, f) = oBind p f
    member x.Return(y) = oReturn y
    member x.ReturnFrom(y) = oReturnFrom y

let option = new OptionMonad()

let isEven x = if x % 2 = 0 then Some x else None 
let minus2 x = Some(x - 2)
let div10 x = if x = 0 then None else Some(10 / x)

Some(4) >>= isEven >>= minus2 >>= div10
Some(5) >>= isEven >>= minus2 >>= div10
Some(2) >>= isEven >>= minus2 >>= div10

let a = option {
    let! a = Some(2)
    let! b = isEven a
    let! c = minus2 b
    let! d = div10 c
    return d
}
let a = option {
    let! a = Some(4)
    let! b = isEven a
    let c = b - 2
    let! d = div10 c
    return d
}

(Some 1) + (Some 2)

option { 
    let! x = Some 1
    let! y = Some 2
    return x + y
} |> printfn "Result 2: %A" 



option.Bind (Some 1, fun x ->
    option.Bind (Some 2, fun y ->
        option.Return (x+y)
    )
) |> printfn "Result 2: %A"
 




let lReturn (x: 'a) : 'a list = [x]

let lReturnFrom (x: 'a list) : 'a list = x

let lBind (mx: 'a list) (f: 'a -> 'b list) : 'b list = 
    List.map f mx |> List.concat
       

let (>>=) = lBind

type ListMonad() =
    member x.Bind(p, f) = lBind p f
    member x.Return(y) = lReturn y
    member x.ReturnFrom(y) = lReturnFrom y

let list = new ListMonad()

[1;2;3] >>= (fun x -> [x; x+1; x+2])

let a = list {
    let! a = [1;2;3]
    let c = 13
    let! b = [4;5;6]
    printf "%A-%A-%A\n" a b c
    // return a + b + c
    return! [a;a+1;a+2]
} 

list.Bind ([1;2;3], fun x ->
    let c = 13
    list.Bind ([4;5;6], fun y ->
        list.Return (x+y+c)
    )
) |> printfn "Result 2: %A"


// writer

let wReturn (x: 'a) : ('a * string) = (x, " got " + x.ToString() + ".\n")
let wReturnFrom (x: 'a * string) : 'a * string = (fst x, snd x + "!\n")
let wBind (mx: ('a * string)) (f: 'a -> 'b * string) : 'b * string = 
    let res = f (fst mx)
    (fst res, snd mx + "\n" + snd res)
                                                       
let (>>=) = wBind

type WriterMonad() =
    member x.Bind(p, f) = wBind p f
    member x.Return(y) = wReturn y
    member x.ReturnFrom(y) = wReturnFrom y

let writer = new WriterMonad()

let squared x = (x * x, " was squared.")
let halved x = (x / 2, " was halved.")
 
let a = writer {
    let! a = wReturn 4
    let! b = squared a
    let! c = halved b
    return c
} // (1, " got 4. was squared. was halved. got 1.")

//////////////////////////////////////////////////////////////

open System
open System.Threading

let ts() = System.DateTime.Now.Ticks

let waitSync id time =
    printfn "%d start" id
    let ts1 = ts()
    Thread.Sleep(time * 1000)
    let ts2 = ts()
    let delta = System.TimeSpan(ts2 - ts1)
    printfn "%d end %s" id (delta.ToString())
    13

[1..10]
  |> List.mapi (fun index time -> waitSync index 1)

let wait id time =
    async {
       printfn "%d start" id
       let ts1 = ts()
       do! Async.Sleep(time * 1000)
       let ts2 = ts()
       let delta = System.TimeSpan(ts2 - ts1)
       printfn "%d end %s" id (delta.ToString("G"))
       return 13
    }

Async.RunSynchronously (wait 1 1)

[1..10]
  |> List.map (fun index -> Async.RunSynchronously (wait index 1))

[1..10]
  |> List.map (fun index -> wait index 1)
  |> Async.Parallel
  |> Async.RunSynchronously

let inside = async {
    let! a = wait 1 1
    let! b = wait 2 1
    return a + b
}
Async.RunSynchronously inside

let inside = async {
    let a = wait 1 1
    let b = wait 2 1
    let [|c; d|] = [a; b] |> Async.Parallel |> Async.RunSynchronously
    return c + d
}
Async.RunSynchronously inside

let inside = async {
    let a = wait 1 1
    let b = wait 2 1
    return! [a; b] |> Async.Parallel
}
Async.RunSynchronously inside

module Async =
  let fmap f workflow = async {
    let! res = workflow
    return f res
  }
  let map f workflow = List.map (fmap f) workflow

[1..10]
  |> List.map (fun index -> wait index 1)
  // |> List.map (Async.fmap (fun x -> x * 2))
  |> Async.map (fun x -> x * 2)
  |> Async.Parallel
  |> Async.RunSynchronously


let calc = 
    [1..10] |> List.map (fun index -> wait index 1)
        //|> List.map (Async.fmap (fun x -> x * 2))
        |> Async.map (fun x -> x * 2)
        |> Async.Parallel

Async.RunSynchronously calc

//////////////////////////////////////////////////

open System.Net
let req1 = HttpWebRequest.Create("http://yandex.ru")
let req2 = HttpWebRequest.Create("http://google.ru")
let req3 = HttpWebRequest.Create("http://bing.com")
req1.BeginGetResponse((fun r1 -> 
    use res1 = req1.EndGetResponse(r1)
    printfn "Downloaded %O" res1.ResponseUri
    req2.BeginGetResponse((fun r2 -> 
        use res2 = req2.EndGetResponse(r2)
        printfn "Downloaded %O" res2.ResponseUri
        req3.BeginGetResponse((fun r3 -> 
            use res3 = req3.EndGetResponse(r3)
            printfn "Downloaded %O" res3.ResponseUri
            ),null) |> ignore
        ),null) |> ignore
    ),null) |> ignore

open System.Net
let req1 = HttpWebRequest.Create("http://yandex.ru")
let req2 = HttpWebRequest.Create("http://google.ru")
let req3 = HttpWebRequest.Create("http://bing.com")

async {
    use! res1 = req1.AsyncGetResponse()  
    printfn "Downloaded %O" res1.ResponseUri
    use! res2 = req2.AsyncGetResponse()  
    printfn "Downloaded %O" res2.ResponseUri
    use! res3 = req3.AsyncGetResponse()  
    printfn "Downloaded %O" res3.ResponseUri
    } |> Async.RunSynchronously

let downloadPage (url: string) = async {
    let req = HttpWebRequest.Create(url)
    use! res = req.AsyncGetResponse()
    printfn "Downloaded %O" res.ResponseUri
}

["http://yandex.ru"; "http://google.ru"; "http://bing.com"]
    |> List.map downloadPage
    |> Async.Parallel
    |> Async.RunSynchronously

#r "../packages/FSharp.Data.2.3.3/lib/net40/FSharp.Data.dll"
open FSharp.Data

let downloadPage (url: string) = async {
    let ts1 = ts()
    let! html = Http.AsyncRequestString(url)
    let ts2 = ts()
    let delta = System.TimeSpan(ts2 - ts1)
    return url, html.Length, delta.ToString("G")
}

["http://yandex.ru"; "http://google.ru"; "http://bing.com"]
    |> List.map downloadPage
    |> Async.Parallel
    |> Async.RunSynchronously

open System.IO
open System.Text
let bases = HtmlDocument.Load("http://mipt.ru/diht/bases/")
//let html = File.ReadAllText("/home/und/fsharp/bases.html")
//let bases = HtmlDocument.Parse(html)
bases.Descendants ["td"]  
    |> Seq.collect (fun (x:HtmlNode) -> x.Descendants ["a"])
    // для получения ссылок вместо InnerText нужно использовать методы TryGetAttribute, Attibute или AttributeValue
    // см. исходный код https://github.com/fsharp/FSharp.Data/blob/master/src/Html/HtmlOperations.fs
    |> Seq.map (fun x -> x.InnerText()) 
    |> Seq.toList
    |> List.filter (fun x -> x <> "")
    
    
let hockey = HtmlDocument.Load("http://www.sport-express.ru/hockey/world/")
hockey.Descendants ["td"]
    |> Seq.filter (fun (td:HtmlNode) -> td.HasClass "t_left w_100p")
    |> Seq.map (fun x -> x.InnerText()) 
    |> Seq.toList    
 
 
