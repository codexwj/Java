**原子性**

`synchronized`和`volatile`两者最大的区别就是在于原子性，`volatile`不具有原子性。原子性就是保证这些操作不可以被中断，要么执行，要么不执行。

**可见性**

可见性是指多个线程访问一个资源时，该资源的状态，值信息等对于其他线程都是可见的。

可以这样理解：当**synchronized**修饰一个类或对象时，一个线程如果要访问该类或对象必须先获取它的锁，而这个锁的状态对于其他线程都是可见。

**有序性**

**可重入性**

`synchronized`和`ReentrantLock`都是可重入锁。当一个线程试图操作一个由其他线程持有的对象锁的临界资源时，将会处于阻塞状态，当一个线程再次请求自己持有对象锁的临界资源时，这种情况属于可重入锁。

`synchronized`可以修饰静态方法、成员函数，同时还可以直接定义代码块。

总的来说上锁的资源只有两类：一个是`对象`，一个是`类`。

要进入同步方法或同步块必须先获得相应的锁才行。

JDK6之后对`synchronized`进行了优化，总共四种状态：**无锁状态，偏向锁，轻量级锁，重量级锁**

https://ask.qcloudimg.com/http-save/yehe-5522483/fu98yi2bmj.webp?imageView2/2/w/1620/format/jpg

对象头是我们关注的重点，它是`synchronized`实现锁的基础，因为`synchronized`申请锁、上锁、释放锁都与对象头有关。对象头主要结构是由`Mark Word`和`Class Metadata Address`组成，其中`Mard Word`存储对象的`hashcode`、锁信息或分带年龄或GC标志信息，`Class Metadata Address`是类型指针指向对象的类元数据，JVM通过该指针确定该对象时那个类的实例。



**不同对象实例的synchronized方法是不相干预的，也就是说，其他线程可以同时访问此类下的另一个对象实例中的synchronized方法**

为了解决上面的局限，可以对类进行加锁，因为类是唯一的。



**对象头包括两部分信息**：

- 自身运行时的数据，比如：锁状态标志，线程持有的锁。
- 类型指针，JVM通过这个指针来确定这个对象时那个类的实例。

**双重校验实现对象单例**

```java
public class Singleton{
    private volatile static Singleton uniqueInstance;
    
    private Singleton(){}
    
    public synchronized static Singleton getUniqueInstance(){
        if(uniqueInstance == null){
            synchronized(Singleton.class){
                if(uniqueInstance == null){
                   uniqueInstance = new Singleton();
                }
            }
        }
    }
    return uniqueInstance;
}
```

```

