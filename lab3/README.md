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
|3|По [списку телефонных номеров МФТИ](http://mipt.ru/about/general/contacts/phones.php) выяснить, кто делит один номер с коллегами. Телефонные номера нормализовать|
|4|По [списку преподавателей](https://mipt.ru/persons/profs/) узнать, кто преподаёт одинаковые дисциплины.|
|5|Составить список из ФИО [преподавателей](http://wikimipt.org/index.php?title=%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%9F%D1%80%D0%B5%D0%BF%D0%BE%D0%B4%D0%B0%D0%B2%D0%B0%D1%82%D0%B5%D0%BB%D0%B8_%D0%BF%D0%BE_%D0%B0%D0%BB%D1%84%D0%B0%D0%B2%D0%B8%D1%82%D1%83) и их страниц в соц.сетях (если есть)|
|6|Узнать, в каком году было больше и меньше всего [выпускников](http://mipt.ru/dafe/graduaters/) факультета аэромеханики и летательной техники|
|7|Найти самый дилнный и самый которкий пример для [Hadoop](https://wiki.apache.org/hadoop) (по кол-ву символов)|
|8|Составить список 50 самых комментируемых [преподавателей](http://wikimipt.org/index.php?title=%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%9F%D1%80%D0%B5%D0%BF%D0%BE%D0%B4%D0%B0%D0%B2%D0%B0%D1%82%D0%B5%D0%BB%D0%B8_%D0%BF%D0%BE_%D0%B0%D0%BB%D1%84%D0%B0%D0%B2%D0%B8%D1%82%D1%83)|
|9|Составить список 50 комментариев для [преподавателей](http://wikimipt.org/index.php?title=%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%9F%D1%80%D0%B5%D0%BF%D0%BE%D0%B4%D0%B0%D0%B2%D0%B0%D1%82%D0%B5%D0%BB%D0%B8_%D0%BF%D0%BE_%D0%B0%D0%BB%D1%84%D0%B0%D0%B2%D0%B8%D1%82%D1%83) с самой высокой оценкой|
|10|Узнать, в каких группах больше всего тем с пометкой "Важно:" на форуме [sql.ru](http://www.sql.ru/forum)|
|11|Узнать, в какой [теме](https://www.opennet.ru/search.shtml?words=Haskell&restrict=forum&sort=score&exclude=) больше всего комментариев|
|12|Узнать, от какой библиотеки зависит больше всего запрещённых [пакетов на hackage](http://hackage.haskell.org/packages/deprecated)|
|13|Найти пять самых скачиваемых пакетов за всё время на [hackage](http://hackage.haskell.org/packages/top)|
|14|Найти пять самых комментируемых [тем](https://archive.codeplex.com/?p=nuget) про NuGet
|15|Попытаться найти по [списку языков программирования](http://en.wikipedia.org/wiki/List_of_programming_languages) википедии человека, который разработал больше всего ЯП|
|16|Сколько пакетов в [Hackage](http://hackage.haskell.org/packages/) относятся к нескольким категориям?
|17|У какого факультета ФИВТ больше всего [кафедр](http://wikimipt.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%9A%D0%B0%D1%84%D0%B5%D0%B4%D1%80%D1%8B_%D0%BF%D0%BE_%D0%B0%D0%BB%D1%84%D0%B0%D0%B2%D0%B8%D1%82%D1%83)?
|18|Узнать, в каком разделе форума [sql.ru](http://www.sql.ru/forum/sqlru-3-days) больше всего вопросов за последние 3 дня|
|19|Узнать, какое количество [вопросов](https://toster.ru/questions), заданных за последние 5 часов, остались без ответа|
|20|Кто, кроме anonymous'а оставил больше всего комментариев в [этой теме](https://www.linux.org.ru/news/google/11404954)?|
|21|Какой тэг самый популярный среди ста самых популярных пакетов на [NuGet](https://www.nuget.org/stats/packages)?
|22|В каком месяце какого года было больше всего вопросов в [рассылке эрланга](http://erlang.org/pipermail/erlang-questions/)?|


