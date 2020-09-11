# Spring Boot知识点

`Spring3.1`中开始对缓存提供支持，核心思路是对方法的缓存，当开发者调用一个方法时，将方法的参数和返回的值作为`key`/`vaule`缓存起来，当再次调用该方法时，如果缓存中有数据，就直接从缓存中获取，否则再去执行该方法。

### `Redis单击缓存`

#### 1.创建项目，添加缓存依赖

创建Spring Boot项目，添加`spring-boot-starter-cache`和`Redis`依赖。

#### 2.缓存配置

`Redis`单机缓存只需要开发者在`application.properties`中进行`Redis`配置及缓存配置即可。

#### 3.开启缓存

在项目入口类中开启缓存。

****

在类上会有`@CacheConfig(cacheNames = "book_cache")`

方法的输入参数作为`key`，返回值作为`value`。

`@Cacheable`：先去缓存里面找，有就直接返回，没有就执行方法。

`@CachePut`：用户数据更新方法上，每次都会去执行方法，缓存里面的都会被覆盖

`@CacheEvict`：用于删除方法上，清楚缓存。

****

----

### `SpringBoot security`安全管理

#### 1.基于内存的认证

如果需要根据实际情况进行角色管理，就需要重写`WebSecurityConfigurerAdapter`中的另一个方法。

----

27-30表示开启表单认证，即读者一开始看到的登录页面，同时配置了登录接口为“login”，即可以直接调用“/login”接口，发起一个POST请求进行登录。

```java
.and()
.formLogin()
.loginProcessingUrl("/login")//方便Ajax或者移动端调用登录接口
.permitAll()//表示和登录相关的接口都不需要认证即可访问
```

由于前后端分离是现在企业级开发的主流，在前后端分离的开发方式中，前后端的数据交互是由`JSON`进行，这是登录成功之后就不是页面跳转了，而是一段`JSON`提示。

----

```java
.loginPage，即登录页面，配置了loginPage后，如果用户未获得授权就去访问一个需要授权才能访问的接口，就会自动跳转到loginPage页面让用户登录。
```

#### 加密方案

密码加密一般会用到散列函数。是一种从任何数据中创建数字“指纹”的方法。

常用的散列函数有：**`MD5`消息摘要算法**、**安全散列算法(Secure Hash Algorithm)**。

在使用散列函数的基础上还需要在密码加密过程中还需要加盐，加盐可以是一个随机数、或者用户名。加盐之后可以实现，即是密码的明文相同的用户生成的密码，密文也不相同。自己配置比较繁琐。

在Spring Security提供了多种密码加密方案，推荐：`BCryptPasswordEncoder`，该方案使用`BCrypt`强哈希函数。

### 基于数据库的认证

在真实项目中，用户的基本信息以及角色等都存储在数据库中。

#### 1.设计数据表

1. 用户表、角色表、用户角色关联表。

#### 2.创建项目

#### 3.配置数据库

```properties
spring.datasource.type = com.alibaba.druid.pool.DruidDataSource
spring.datasource.username = root
spring.datasource.password = root
spring.datasource.url = jdbc:mysql://
```

#### 4.创建实体类

#### 5.创建UserService

#### 6.配置Spring Security

-----

### 动态配置权限

为了实现资源和角色之间的动态调整，要实现动态配置URL权限。就需要开发者自定义权限配置。



### WebSocket简介

WebSocket是一种在单个TCP连接上进行全双工通信的协议。只需一次握手就可以直接创建持久性的连接，并进行双向数据传输。

Spring框架提供了基于WebSocket的STOMP支持，STOMP是一个简单的可互操作的协议，通常被用于中间服务器和客户端之间进行异步消息传递。

### JMS

（Java Message Service）包括两种消息模型：点对点和发布者/订阅者。

Spring Boot整合JMS

`RabbitMQ`是一个实现了`AMQP`的开源消息中间件。

**在`RabbitMQ`中，消息生产者提交的消息都会交由`exchange`进行在分配，`Exchange`会根据不同的策略将信息分发到不同的`Queue`中。**

`DirectExchange`的路由策略是将消息队列绑定到一个`DirctExchange`上，当一条消息到达`DirectExchange`时会被转发到与该条消息`routing key`相同的`Queue`上。

`FanoutExchange`的路由策略是将消息队列绑定到一个`FanoutExchange`上，当消息叨叨`FanoutExchange`时会被转发给与它绑定的`Queue`。

在`TopicExchange`中，`Queue`通过`routingkey`绑定到`TopicExchange`上，当消息到达`TopicExchange`后，`TopicExchange`根据消息的`routingkey`将消息路由到一个或者多个`Queue`上。