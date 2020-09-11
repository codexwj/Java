## Java基础知识点

- 把`JSON`解析为`JavaBean`的过程称为反序列化。把`JavaBean`变成`JSON`，就是序列化。
- 基于表（`table`）一对多的关系是关系数据库的基础。
- `JDBC`(`Java Database Connectivity`)，Java程序访问数据的标准接口。
- `JDBC`驱动就是一个第三方库。如：`mysql-connector-java`
- `JDBC`连接：
  
- URL：`jdbc:mysql://<hostname>:<port>/<db>?key1=value1&key2=value2`
  
- `数据库事务` ：是由若干个SQL语句构成的一个操作序列。
- 把同一个SQL但参数不同的若干操作合并为一个batch执行。
- 由于每次创建JDBC连接和销毁JDBC连接CPU的开销太大，可以通过一个连接池(Connection Pool)复用已经创建好的连接。
- 注意：`Mybatis`执行查询后，将根据方法的返回类型自动把`ResultSet`的每一行转换为User实例，转化的规则就是按照列名和属性名对应。如果列名和属性名不同，最简单的方式是编写`SELECT`语句的别名。

- .class是JVM可以执行的最小文件，一个大型程序需要很多class，并生成一堆的.class文件，很不便于管理，所有jar就是class文件的容器，但是jar并不关心class文件之间的依赖。因此，在Java9开始引入的模块，就是为了解决class文件之间的依赖关系。模块的后缀`.jmod`

  可以总结为：jar只是进行打包封装，但是模块在jar的基础上还实现了写入依赖关系。

- `String类型`两个字符串比较必须使用`equals()`方法。

  常见的方法：`contains()`，`indexOf()`, `lastIndexOf()`, `startsWith()`, `endsWith()`，`substring()`, `isEmpty()`, `isBlank()`。

  **去除收尾空白字符**：`.trim()` ：并没有改变字符串的内容，而是返回了一个新的字符串。

  `.strip()`：方法也可以移除字符串首尾空白字符，但是`\u3000`也会被移除。

  **替换子串**：`replace(,)`

  ```java
  String s = "hello";
  s.replace('l','w'); //"hewwo"
  
  //正则表达式
  String s = "A,,B;C ,D";
  s.replaceAll("[\\,\\;\\s]+", ",");//"A,B,C,D"
  ```

  **分割字符串**：`split()`，传入的也是正则表达式：

  ```java
  String s = "A,B,C,D";
  String[] ss = s.split("\\,");// {"A","B","C","D"}
  ```

  **拼接字符串**：使用静态方法`join()`，使用指定的字符串连接字符串数组。

  ```java
  String[] arr = {"A","B","C","D"};
  String s = String.join("***",arr);//"A***B***C***D"
  ```

  **格式化字符串**：`formatted()`和`format()静态方法`，可以传入其他参数，替换占位符，然后生成新的字符串。

  ```java
   String s = "Hi %s, your score id %d!";
  sout(s.formatted("Alice",80));
  %s
  %d
  %x
  %f
  ```

  **类型转换**

  `String.valueOf()`

  `Integer.parseInt("124")`

  `Boolean.parseBoolean("true")`

  **转换为char[]**

  相互转换：

  ```java
  char[] cs = "Hello".toCharArray();
  String s = new String(cs);
  ```

  转换编码就是将`String`和`byte[]`转换。


- 包装类型：把基本类型(如int)变为引用类型(如Integer)的赋值写法，称为自动装箱(Auto Boxing)，反过来称为自动拆箱(Auto Unboxing)，但是互相变化为影响效率，一般不会进行转换。
- **不变类**
- Unicode和UTF-8区别：Unicode是字符集,即字符集：为每一个字符分配一个唯一的ID（学名码位/码点/Code Point)。UTF-8是编码规则：将码位转换为字节序列的规则。

----

```xml
Unicode字符集为每个字符分配一个码位，例如“知”的码位是30693,表示为16进制为（ox77e5）
UTF-8是一套以8位为一个编码单位的可变长编码，会将一个码位编码为1到4个字节。
UTF-8 就是使用变长字节表示,顾名思义，就是使用的字节数可变，这个变化是根据 Unicode 编号的大小有关，编号小的使用的字节就少，编号大的使用的字节就多。使用的字节个数从 1 到 4 个不等。
由于 UTF-8 的处理单元为一个字节（也就是一次处理一个字节），所以处理器在处理的时候就不需要考虑这一个字节的存储是在高位还是在低位，直接拿到这个字节进行处理就行了，因为大小端是针对大于一个字节的数的存储问题而言的。
```



-----