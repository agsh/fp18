## Lab3
### Описание

В этой лабораторной вам предстоит работать со списками, полученными в результате разбора веб-страниц.
Для этих целей используются библиотеки [FSharp.Data](https://fsharp.github.io/FSharp.Data/library/HtmlParser.html) и [*-conduit](https://github.com/snoyberg/xml). Они позволяют изящно разбирать и запрашивать данные из таких иерархических структур как json, xml, html. В нашем случае мы будем работать с html-страницами, запрашивать содержимое тэгов с помощью селекторов и получать нужные нам данные в последовательностях и списках соответственно.

Примеры работы с библиотеками для каждого из языков находятся в соответствующих boilerplate-файлах: [Lab3.fs](./Lab3.fs) и [Lab3.hs](./Lab3.hs). И в интернете: [FSharp.Data](https://fsharp.github.io/FSharp.Data/library/HtmlParser.html), [html-conduit](https://www.fpcomplete.com/school/starting-with-haskell/libraries-and-frameworks/text-manipulation/tagsoup)

### Установка библиотек
#### FSharp
Через NuGet: `Install-Package FSharp.Data `. Или через nuget-плагин для monodevelop.
#### Haskell
С использованием cabal: `cabal install xml-conduit http-conduit html-conduit`

### Вопросы
Вполне вероятно, что некоторые страницы будут парситься некорректно, данные будет сложно извлечь, возникнут проблемы с кодировкой. Все эти и другие вопросы, как обычно, задавайте в issue.

### Задание
|Вариант|Задание|
|---|---|
|1|По [списку языков программирования](http://en.wikipedia.org/wiki/List_of_programming_languages) википедии составить список императивных, не функциональных ЯП.|
|2|По [списку языков программирования](http://en.wikipedia.org/wiki/List_of_programming_languages) википедии составить список кортежей: год  появления, названия. Языки без указания годов появления исключить.|
