module lab0

open System
open System.Net
open System.Collections.Specialized

let (email, name) = ("", "") // Адрес почты и фамилия с инициалами
let university = "" // ВУЗ ( MEPhI | MIPT )
let group = "" // Группа

let pascal c r = 1 // а тут решение

let printIt n = 
  "[" +
  ([for x in 0..n do for y in 0..x do yield pascal y x] 
    |> List.map (fun x -> x.ToString())
    |> List.reduce (fun x y -> x + "; " + y) )
  + "]"

let main () = 
  let values = new NameValueCollection()
  values.Add("email", email)
  values.Add("name", name)
  values.Add("university", university)
  values.Add("group", group)
  values.Add("content", printIt 20)

  let client = new WebClient()
  let response = client.UploadValues(new Uri("http://91.239.142.110:13666/lab0"), values)
  let responseString = Text.Encoding.Default.GetString(response)

  printf "%A\n" responseString
