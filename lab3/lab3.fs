// подключаем FSharp.Data
#r "../packages/FSharp.Data.2.3.3/lib/net40/FSharp.Data.dll"
open FSharp.Data
open System
open System.IO
open System.Net
open System.Text
open System.Collections.Specialized

// почтовый адрес
let email = ""

let lab3 () =
  let bases = HtmlDocument.Load("http://mipt.ru/diht/bases/")
  bases.Descendants ["ul"] 
    |> Seq.filter (fun x -> x.HasClass("right-menu")) 
    |> Seq.collect (fun (x:HtmlNode) -> x.Descendants ["a"])
    // для получения ссылок вместо InnerText нужно использовать методы TryGetAttribute, Attibute или AttributeValue
    // см. исходный код https://github.com/fsharp/FSharp.Data/blob/master/src/Html/HtmlOperations.fs
    |> Seq.map(fun x -> x.InnerText()) 
    |> Seq.toList

let main () = 
  let values = new NameValueCollection()
  values.Add("email", email)
  values.Add("result", lab3().ToString())
  values.Add("content", File.ReadAllText(__SOURCE_DIRECTORY__ + @"/" + __SOURCE_FILE__))

  let client = new WebClient()
  let response = client.UploadValues(new Uri("http://91.239.142.110:13666/lab3"), values)
  let responseString = Text.Encoding.Default.GetString(response)

  printf "%A\n" responseString
