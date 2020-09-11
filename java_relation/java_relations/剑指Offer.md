# 剑指Offer

### 不用加减乘除做加法（位运算）

```java
class Solution{
    public int add(int a, int b){
        while(b!=0){
            int c = (a&b)<<1;//进行进位，向前进一个
            int a ^= b;//异或相同为0，不同为1
            b = c;
        }
        return a;
    }
}
```

将无进位和、进位分开

$\begin{cases} \ n=a\bigoplus b\\ c=a\&b<<1 \end{cases}$

### 翻转单词顺序 

```java
public class Solution{
    public String reverseWords(String s){
        StringBuilder sb = new StringBuilder();
        String[] str = s.trim().split(" ");
        for(int i=str.length-1;i>=0;i--){
            if(str[i].equals(" ")) continue;
            sb.append(str[i] + " ");
        }
        return sb.toString().trim();
    }
}
```

### 数组中数字出现的次数||，只有一个出现一次，其他出现了三次

出现了三次的数字二级制每一位之和是3的倍数。

```java
class Solution{
    public int singleNumber(int[] nums){
        int[] counts = new int[32];
        for(int num:nums){
            for(int j=0;j<32;j++){
                counts[j] += num & 1;
                num >>>= 1;
            }
        }
        int res = 0, m = 3;
        for(int i=0;i<32;i++){
            res <<= 1;
            res |= counts[31 - i] % m;
        }
        return res;
    }
}
```

### 数组中数字出现的次数

一个整型数组 `nums` 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是$O(n)$，空间复杂度是$O(1)$。

```java
class Solution{
    public int[] singleNumbers(int[] nums){
        int xorNumber = nums[0];
        for(int i=1;i<nums.length;i++){
            xorNumber ^= nums[i];
        }
        int onePosition = xorNumber&(-xorNumber);
        int ans1 = 0, ans2 = 0;
        for(int i=0;i<nums.length;i++){
            if((onePosition&nums[i]) == onePosition){
                ans1 ^= nums[i];
            }else{
                ans2 ^= nums[i];
            }
        }
        return new int[]{ans1^0, ans2^0};
    }
}
```

### 判断平衡二叉树

**方法一**：后序遍历+剪枝（从底至顶）

对二叉树进行后序遍历，从底至顶返回子树的深度，如判定某子树不是平衡树，则剪枝，直接向上返回。

```java
class Solution{
    public boolean isBalanced(TreeNode root){
        return recur(root) != -1;
    }
    private int recur(TreeNode root){
        if(root == null) return 0;
        int left = recur(root.left);
        if(left == -1) return -1;
        int right = recur(root.right);
        if(right == -1) return -1;
        return Math.abs(left - right) < 2 ? Math.max(left,right) + 1:-1;
    }
}
```

### 二叉搜索树的第K大结点，中序遍历倒序为递减序列

```java
class Solution{
    int res, k;
    public int kthLargest(TreeNode root, int k){
        this.k = k;
        dfs(root);
        return res;
    }
    
    void dfs(TreeNode root){
        if(root == null) return;
        dfs(root.right);
        if(k == 0) return;
        if(--k == 0) res = root.val;
        dfs(root.left);
    }
}
```

### 0~n-1中缺失的数字

**二分**：

```java
public int missingNumber(int[] nums){
    int left = 0;
    int right = nums.length;
    while(left < right){
        int mid = (left + right) / 2;
        if(nums[mid] != mid) right = mid;
        else{
            left = mid + 1;
        }
    }
    return left;
}
```

### 在排序数组中查找数字|

统计一个数字在排序数组中出现的次数。

两个右边界相减

```java
class Solution{
    public int search(int[] nums,int target){
        return helper(nums, target) - helper(nums, target - 1);
    }
    
    int helper(int[] nums, int target){
        int i = 0, j = nums.length - 1;
        while(i <= j){
            int m = (i + j) / 2;
            if(nums[m] <= target) i = m + 1;
            else j = m-1;
        }
        if(i >= right || nums[i]!= target) return -1;
        return i;
    }
}
```

### 数组中的逆序对

```java
class Solution{
    public int reversePairs(int[] nums){
        int len = nums.length;
        if(len == 0) return 0;
        return mergeSort(nums,0,len-1);
    }
    
    public int mergeSort(int[] nums, int left, int right){
        if(left >= right) return 0;
        int mid = (right + left) >> 1;
        int l = mergeSort(nums,left,mid);
        int r = mergeSort(nums,mid+1,right);
        return (l+r+merge(nums,left,mid,right));
    }
    
    public int merge(int[] nums,int left, int mid, int right){
        int[] temp = new int[right-left+1];
        int ans = 0;
        int cur1 = mid;
        int cur2 = right;
        int cur3 = right - left;
        while(cur1 >= left&&cur2 >= mid+1){
            if(nums[cur1]<=nums[cur2]){
                temp[cur3--] = nums[cur2--];
            }else{
                temp[cur3--] = nums[cur1--];
                ans += cur2-mid;
            }
        }
        while(cur1 >= left){
            temp[cur3--] = nums[cur1--];
        }
        while(cur2 >= mid+1){
            temp[cur3--] = nums[cur2--];
        }
        int x = 0;
        while(left<=right){
            nums[left++]=temp[x++];
        }
        return ans;
    }
}
```

自己更能理解的

```java
public int reversePairs(int[] nums) {
     return merge(nums, 0, nums.length - 1);
}

int merge(int[] arr, int start, int end) {
    if (start == end) return 0;
     int mid = (start + end) / 2;
     int count = merge(arr, start, mid) + merge(arr, mid + 1, end);

     int[] temp = new int[end - start + 1];
     int i = start, j = mid + 1, k = 0;
     while (i <= mid && j <= end) {
     count += arr[i] <= arr[j] ? j - (mid + 1) : 0;
        temp[k++] = arr[i] <= arr[j] ? arr[i++] : arr[j++];
     }
     while (i <= mid) {
        count += j - (mid + 1);
        temp[k++] = arr[i++];
     }
     while (j <= end)
         temp[k++] = arr[j++];
     System.arraycopy(temp, 0, arr, start, end - start + 1);
     return count;
}
```

### 礼物的最大价值

**思路：**礼物的最大价值，

```java
class Solution{
    public int maxValue(int[][] grid){
        int m = grid.length, n = grid[0].length;
        for(int i = 0;i < m;i++){
            for(int j = 0;j < n;j++){
                if(i == 0 && j == 0) continue;
                if(i == 0) grid[i][j] += grid[i][j-1];
                else if(j == 0) grid[i][j] += grid[i-1][j];
                else grid[i][j] += Math.max(grid[i-1][j], grid[i][j-1]);
                
            }
        }
        return grid[m-1][n-1];
    }
}
```

```java
class Solution{
    public int maxValue(int[][] grid){
        int m = grid.length, n = grid[0].length;
        for(int i=1;i<m;i++){
            grid[i][0] += grid[i-1][0];
        }
        for(int j = 1;j < n;i++){
            grid[0][j] += grid[0][j-1];
        }
        for(int i=1;i<m;i++){
            for(int j=1;j<n;j++){
                grid[i][j] += Math.max(grid[i-1][j], grid[i][j-1]);
            }
        }
        return grid[m-1][n-1];
    }
}
```

```java
//多开一行一列能使代码更加简洁
class Solution{
    public int maxValue(int[][] grid){
        int row = grid.length;
        int column = grid[0].length;
        
        int[][] dp = new int[row+1][column+1];
        for(int i=1;i<= row;i++){
            for(int j=1;j<=column;j++){
                dp[i][j] = Math.max(dp[i-1][j],dp[i][j-1]) + grid[i-1][j-1];
            }
        }
        return dp[row][column];
    }
}
```



### 把数字翻译成字符串

动态规划：字符串遍历

```java
class Solution{
    public int translateNum(int num){
        String s = String.valueOf(num);
        int a = 1, b = 1;
        for(int i=2;i<=s.length();i++){
            String tmp = s.substring(i - 2,i);
            int c = tmp.compareTo("10") >= 0 && tmp.compareTo("25") <=0? a+b:b;
            a = b;
            b = c;
        }
        return b;
    }
}
```

更直观的方法：

```java
class Solution{
    public int translateNum(int num){
        String s = String.valueOf(num);
        int[] dp = new int[s.length() + 1];
        dp[0] = 1;
        dp[1] = 1;
        for(int i=2;i<=s.length();i++){
            String temp = s.substring(i-2,i);
            if(temp.compareTo("10") >= 0 && temp.compateTo("25") <= 0){
                dp[i] = dp[i-1]+dp[i-2];
            }else{
                dp[i] = dp[i-1];
            }
        }
        return dp[s.length()];
    }
}
```



### 把数组排成最小的数

本质上是一个排序问题

**示列**

```java
输入: [10,2]
输出: "102"
```



```java
class Solution{
    public String minNumber(int[] nums){
        String[] strs = new String[nums.length];
        for(int i=0;i<nums.length;i++){
            strs[i] = String.valueOf(nums[i]);
        }
        fastSort(strs, 0, strs.length - 1);
        StringBuilder res = new StringBuilder();
        for(String s: strs){
            res.append(s);
        }
        return res.toString();
    }
    private void fastSort(String[] strs, int l, int r){
        if(l >= r) return;
        int i=l, j = r;
        String tmp = strs[i];
        while(i < j){
            while((strs[j] + strs[l]).compareTo((strs[l] + strs[j]))>=0 && i < j) j--;
            while((strs[i] + strs[l]).compareTo((strs[l] + strs[i])) <= 0 && i < j) i++;
            tmp = strs[i];
            strs[i] = strs[j];
            strs[j] = tmp;
        }
        strs[i] = strs[l];
        strs[l] = tmp; 
        fastSort(strs, l, i-1);
        fastSort(strs, i+1, r);
    }
}
```

### 数字序列中某一位的数字

数字以0123456789101112131415…的格式序列化到一个字符序列中。在这个序列中，第5位（从下标0开始计数）是5，第13位是1，第19位是4，等等。

`思路：`

- 确定 `n`所在 数字 的 位数 ，记为 `digit` ；
- 确定 `n `所在的 数字 ，记为 `num `；
- 确定` n`是` num`中的哪一数位，并返回结果。

```java
class Solution{
    //n是位数
    public int findNthDigit(int n){ //n称为数位
        int digit = 1; //位数，比如10表示两位数
        long start = 1;//每digit位数的起始数字
        long count = 9;
        while(n > count){
            n -= count;
            digit += 1;
            start *= 10;
            count = digit * start * 9;
        }
        long num = start + (n - 1) / digit;
        return Long.toString(num).charAt((n-1) % digit) - '0';
    }
}
```

### 1~n整数中1出现的次数

怎样去理解：生活中常见的密码锁，就是**那种几个滚轮的密码锁，固定其中的一位密码，拨动其他位置的密码**。

```java
class Solution{
    public int countDigitOne(int n){
        int digit = 1, res = 0;
        int high = n/10, cur = n%10, low = 0;
        while(high != 0 || cur != 0){
            if(cur == 0) res += high * digit;
            else if(cur == 1) res += high * digit + low + 1;
            else res += (high + 1) * digit;
            low += cur * digit;
            cur = high % 10;
            high /= 10;
            digit *= 10;
        }
        return res;
    }
}
```

### 连续子数组的最大和

**思路**：动态规划

```java
class Solution{
    public int maxSubArray(int[] nums){
        int res = nums[0];
        for(int i=1;i<nums.length;i++){
            nums[i] += Math.max(nums[i-1],0);
            res = Math.max(res, nums[i]);
        }
        return res;
    }
}
```

### 数据流中的中位数

建立一个小顶堆和一个大顶堆

```java
class MedianFinder{
    Queue<Integer> A,B;
    public MediaFinder(){
        A = new PriorityQueue<>();
        B = new PriorityQueue<>((x,y)->(y-x));
    }
    public void addNum(int num){
        if(A.size() != B.size()){
            A.add(num);
            B.add(A.poll());
        }else{
            B.add(num);
            A.add(B.poll());
        }
    }
    public double findMedian(){
        return A.size() != B.size() ? A.peek():(A.peek() + B.peek()) / 2.0;
    }
}
```

### 数组中出现次数超过一半的数字（摩尔投票法）

```java
class Solution{
    public int majorityElement(int[] nums){
        int x = 0, votes = 0;
        for(int num : nums){
            if(votes == 0) x = num;
            votes += num == x ? 1 : -1;
        }
        return x;
    }
}
```

### 字符串的排列

```java
class Solution{
    List<String> res = new LinkedList<>();
    char[] c;
    public String[] permutation(String s){
        c = s.toCharArray();
        dfs(0);
        return res.toArray(new String[res.size()]);
    }
    
    void dfs(int x){
        if(x == c.length - 1){
            res.add(String.valueOf(c));
            return;
        }
        HashSet<Character> set = new HashSet<>();
        for(int i = x;i < c.length;i++){
            if(set.contains(c[i])) continues;
            set.add(c[i]);
            swap(i,x);
            dfs(x + 1);
            swap(i,x);
        }
    }
    void swap(int a, int b){
        char tmp = c[a];
        c[a] = c[b];
        c[b] = tmp;
    }
}
```

### 序列化二叉树

请实现两个函数，分别用来序列化和反序列化二叉树

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class Codec {
    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        if(root == null) return new String("[]");
        StringBuilder res = new StringBuilder("[");
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while(!queue.isEmpty()){
            TreeNode node = queue.poll();
            if(node != null){
                res.append(node.val + ",");
                queue.add(node.left);
                queue.add(node.right);
            }else{
                res.append("null,");
            }
        }
        res.deleteCharAt(res.length() - 1);
        res.append("]");
        return res.toString();
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if(data.equals("[]")) return null;
        String[] vals = data.substring(1,data.length() - 1).split(",");
        TreeNode root = new TreeNode(Integer.valueOf(vals[0]));
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int i = 1;
        while(!queue.isEmpty()){
            TreeNode node = queue.poll();
            if(!vals[i].equals("null")){
                node.left = new TreeNode(Integer.parseInt(vals[i]));
                queue.add(node.left);
            }
            i++;
            if(!vals[i].equals("null")){
                node.right = new TreeNode(Integer.parseInt(vals[i]));
                queue.add(node.right);
            }
            i++;
        }
        return root;
    }
}
```

### 二叉搜索树与双向链表

输入一颗二叉搜索树，将该二叉树转换为一个排序的循环双向链表。要求不能创建任何新的结点，只能调整树中接地那指针的指向。

```java
/*
class Node{
	public int val;
	public Node left;
	public Node right;
	
	public Node(){
		
	}
	public Node(int _val){
		val = _val;
	}
	public Node(int _val,Node _left,Node _right){
		val = _val;
		left = _left;
		right = _right;
	}
}
*/
class Solution{
    Node pre,head;
    public Node treeToDoublyList(Node root){
        if(root == null) return null;
        dfs(root);
        head.left = pre;
        pre.right = head;
        return head;
    }
    void dfs(TreeNode root){
        if(root == null) return;
        dfs(root.left);
        if(pre != null) pre.right = root;
        else head = root;
        root.left = pre;
        pre = root;
        dfs(root.right);
    } 
}
```

### 复杂链表的复制

HashMap

```java
class Solution{
    public Node copyRandomList(Node head){
        HashMap<Node, Node> map = new HashMap<>();
        Node cur = head;
        while(cur != null){
            map.put(cur,new Node(cur.val));
            cur = cur.next;
        }
        cur = head;
        while(cur != null){
            map.get(cur).next = map.get(cur.next);
            map.get(cur).random = map.get(cur.random);
            cur = cur.next;
        }
        return map.get(head);
    }
}
```



```java
class Solution {
    LinkedList<List<Integer>> res = new LinkedList<>();
    LinkedList<Integer> path = new LinkedList<>(); 
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        recur(root, sum);
        return res;
    }
    void recur(TreeNode root, int tar) {
        if(root == null) return;
        path.add(root.val);
        tar -= root.val;
        if(tar == 0 && root.left == null && root.right == null)
            res.add(new LinkedList(path));
        recur(root.left, tar);
        recur(root.right, tar);
        path.removeLast();
    }
}
```

### 第一个只出现一次的字符（有序哈希表）

```java
class Solution{
    public char firstUniqChar(String s){
        Map<Character, Boolean> map = new HashMap<>();
        char[] sc = s.toCharArray();
        for(char c:sc){
            map.put(c, !dic.containsKey(c));
        }
        for(Map.Entry<Character,Boolean> d:dic.entrySet()){
            if(d.getValue()) return d.getKey();
        }
        return ' ';
    }
}
```

### 丑数

```java
class Solution{
    public int nthUglyNumber(int n){
        int a = 0, b = 0, c = 0;
        int[] dp = new int[n];
        dp[0] = 1;
        for(int i=1;i<n;i++){
            int n2 = dp[a] * 2, n3 = dp[b] * 3, n5 = dp[c] * 5;
            dp[i] = Math.min(Math.min(n2,n3),n5);
            if(dp[i] == n2) a++;
            if(dp[i] == n3) b++;
            if(dp[i] == n5) c++;
        }
        return dp[n-1];
    }
}
```

### 验证搜索树的后序遍历序列（递归分治）

### 二叉树的后序遍历序列

```java
class Solution{
    public boolean verifyPostorder(int[] postorder){
        return recur(postorder,0,postorder.length - 1);
    }
    boolean recur(int[] postorder,int i, int j){
        if(i >= j) return true;
        int p = i;
        while(postorder[p] < postorder[j]) p++;
        int m = p;
        while(postorder[p] > postorder[j]) p++;
        return p == j && recur(postorder,i,m-1) && recur(postorder,m,j-1);
    }
}
```

### 从上到下打印二叉树|||（BFS/双端队列）

```java
class Solution{
    public List<List<Integer>> levelOrder(TreeNode root){
        Queue<TreeNode> queue = new LinkedList<>();
        List<List<Integer>> res = new LinkedList<>();
       	if(root != null) queue.add(root);
        while(!queue.isEmpty()){
            LinkedList<Integer> tmp = new LinkedList<>();
            for(int i = queue.size();i > 0;i--){
                TreeNode node = queue.poll();
                if(res.size() % 2 == 0) tmp.addLast(node.val);
                else{
                    tmp.addFirst(node.val);
                }
                if(node.left != null) queue.add(node.left);
                if(node.right != null) queue.add(node.right);
            }
            res.add(tmp);
        }
        return res;
    }
}
```

### Shell排序

```java
class Solution{
    public shellSort(int[] arr){
        for(int gap = arr.length / 2;gap >=0;gap /= 2){
            for(int i = gap; i<arr.length;i++){
                int j = i;
                int temp = arr[j];
                if(arr[j] < arr[j - gap]){
                    while(j-gap >= 0 && temp < arr[j - gap]){
                        arr[j] = arr[j - gap];
                        j -= gap;
                    }
                    arr[j] = temp;
                }
            }
        }
    }
}
```

### 计数排序

```java
public static int[] countSort(int[] nums){
    int max = Integer.MIN_VALUE;
    for(int num : nums){
        if(num > max){
            max = num;
        }
    }
    int[] count = new int[max + 1];
    for(int num : nums){
        count[num]++;
    }
    int[] result = new int[nums.length];
    int index = 0;
    for(int i=0;i<max;i++){
        while(count[i] > 0){
            result[index++] = i;
            count[i]--;
        }
    }
    return result;
}
```

### 生产者和消费者

利用synchronized实现

```java
import java.util.LinkedList;

public class Main {
    //常量
    private static int MAX_VALUE = 100;
    public static void main(String[] args) {
        Concurrentcomm con = new Concurrentcomm();
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    for (int i= 0;i < MAX_VALUE;i++){
                        Thread.sleep(0);
                        con.product();
                    }
                }catch (Exception e){
                    e.printStackTrace();
                }
            }
        }).start();
        //消费者
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    Thread.sleep(1000);
                    for (int i=0;i<MAX_VALUE;i++){
                        con.customer();
                    }
                }catch (Exception e){
                    e.printStackTrace();
                }
            }
        }).start();
    }
}

class Concurrentcomm{
    //常量
    private static int MAX_VALUE = 10;
    //缓存
    LinkedList<String> linkedList = new LinkedList<>();
//    Object object = new Object();

    /**
     * 生产者方法
     * @throws Exception
     */
    public void product() throws Exception{
        synchronized (linkedList){
            while (MAX_VALUE == linkedList.size()){
                System.out.println("仓库已满,【生产者】：暂时不能执行生成任务！");
                linkedList.wait();
            }
            linkedList.push("李四");
            System.out.println("【生产者】：生产了一个产品\t【现仓储量为】：" + linkedList.size());
            linkedList.notify();
        }
    }

    public void customer() throws Exception{
        /**
         * 根据jdk的void notifyAll()的描述，“解除那些在该对象上调用wait()方法的线程的阻塞状态。该方法只能在同步方法或同步块内部调用。
         * 如果当前线程不是对象所得持有者，
         * 该方法抛出一个java.lang.IllegalMonitorStateException异常
         * 因此，我们使用同一把锁
         */
        synchronized (linkedList){
            //多线程判断中使用while不要使用if否则会出现虚假唤醒问题
            while(linkedList.size() == 0){
                System.out.println("仓库无货，【消费者】：暂时不能执行消费任务！");
                linkedList.wait();
            }
            linkedList.poll();
            System.out.println("【消费者】：消费了一个产品\t【现仓储量为】：" + linkedList.size());
            linkedList.notify();
        }
    }
}
```



利用阻塞队列实现

```java
//生产者
class Producer implements Runnable{
    //共享阻塞队列
    private BlockingDeque<Data> queue;
    //是否还在运行
    private volatile boolean isRunning = true;
    //id生成器
    private static AtomicInteger count = new AtomicInteger();
    //生成随机数
    private static Random random = new Random();

    public Producer(BlockingDeque<Data> queue){
        this.queue = queue;
    }

    @Override
    public void run() {
        try {
            while(isRunning){
                //模拟注水耗时
                Thread.sleep(random.nextInt(1000));
                int num = count.incrementAndGet();
                Data data = new Data(num, num);
                System.out.println("当前>>注水管:"+Thread.currentThread().getName()+"注水容量(L):"+num);
                if(!queue.offer(data,2, TimeUnit.SECONDS)){
                    System.out.println("注水失败...");
                }
            }
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    public void stop(){
        isRunning = false;
    }
}

//消费者
class Consumer implements Runnable{

    private BlockingDeque<Data> queue ;

    private static Random random = new Random();

    public Consumer(BlockingDeque<Data> queue){
        this.queue = queue;
    }

    @Override
    public void run() {
        while (true){
            try {
                Data data = queue.take();
                //模拟抽水耗时
                Thread.sleep(random.nextInt(1000));
                if(data != null){
                    System.out.println("当前<<抽水管:"+Thread.currentThread().getName()+",抽取水容量(L):"+data.getNum());
                }
            }catch (Exception e){
                e.printStackTrace();
            }

        }
    }
}

//生产的数据
/**
 * 建立一个模拟数据，水
 */
class Data {
    private int id;
    private int num;

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public int getNum() {
        return num;
    }

    public void setNum(int num) {
        this.num = num;
    }

    public Data(int id, int num) {
        this.id = id;
        this.num = num;
    }

    public Data() {

    }
}

import java.util.Random;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class Main {
    public static void main(String[] args) throws InterruptedException {

        BlockingDeque<Data> queue = new LinkedBlockingDeque<>(10);

        Producer producer1 = new Producer(queue);
        Producer producer2 = new Producer(queue);
        Producer producer3 = new Producer(queue);

        Consumer consumer1 = new Consumer(queue);
        Consumer consumer2 = new Consumer(queue);
        Consumer consumer3 = new Consumer(queue);

        ExecutorService service = Executors.newCachedThreadPool();
        ExecutorService service1 = Executors.newFixedThreadPool(2);
//        service.execute(producer1);
//        service.execute(producer2);
//        service.execute(producer3);
        service1.execute(producer1);
        service1.execute(consumer1);


//        service.execute(consumer1);
//        service.execute(consumer2);
//        service.execute(consumer3);

        Thread.sleep(3000);
        producer1.stop();
//        producer2.stop();
//        producer3.stop();

        Thread.sleep(1000);
        service.shutdown();
    }
}
```

### 验证栈的压入、弹出序列

```java
class Solution{
    public boolean validateStackSequences(int[] pushed, int[] popped){
        LinkedList<Integer> stack = new LinkedList<>();
        int i = 0;
        for(int num : pushed){
            stack.push(num);
            while(!stack.isEmpty() && stack.peek() == popped[i]){
                stack.pop();
                i++;
            }
        }
        return stack.isEmpty();
    }
}
```

### 包含min函数的栈（辅助栈，清晰图解）

```java
class MinStack{
    Stack<Integer> A,B;
    public MinStack(){
        A = new Stack<>();
        B = new Stack<>();
    }
    public void push(int x){
        A.add(x);
        if(B.empty() || B.peek() >= x){
            B.add(x);
        }
    }
    public void pop(){
        if(A.pop().equals(B.peek())){
            B.pop();
        }
    }
    publit int top(){
        return A.peek();
    }
    public int min(){
        return B.peek();
    }
}
```

### 顺时针打印矩阵

```java
class Solution{
    public int[] spiralOrder(int[][] matrix){
        if(matrix == null) return new int[0];
        int l= 0, r = matrix[0].length - 1,t=0,b = matrix.length - 1,x = 0;
        int[] res = new int[(r + 1)*(b+1)];
        while(true){
            for(int i=l;i<=r;i++) res[x++] = matrix[t][i];
            if(++t > b) break;
            for(int i=t;i<=b;i++) res[x++] = matrix[i][r];
            if(l > --r) break;
            for(int i=r;i>=l;i--) res[x++] = matrix[b][i];
            if(t > --b) break;
            for(int i=b;i>=t;i--) res[x++] = matrix[i][l];
            if(++l > r) break;
        }
        return res;
    }
}
```

### 对称二叉树（递归）

```java
class Solution{
    public boolean isSymmetric(TreeNode root){
        return isMirror(root,root);
    }
    public boolean isMirror(TreeNode A,TreeNode B){
        if(A == null && B == null) return true;
        if(A == null || B == null) return false;
        return A.val == B.val && isMirror(A.left,B.right) && isMirror(A.right, B.left);
    }
}
```

### 二叉树的镜像

请完成一个函数，输入一个二叉树，该函数输出它的镜像。左右结点交换。

```java
class Solution{
    public TreeNode mirrorTree(TreeNode root){
        if(root == null) return root;
        TreeNode Node = mirrorTree(root.left);
        root.left = mirrorTree(root.right);
        root.right = Node;
        return root;
    }
}
```

### 是否是树的子结构

```java
class Solution{
    public boolean isSubStructure(TreeNode A,TreeNode B){
        return (A!=null && B!=null) && (recur(A,B) || isSubstructure(A.left,B) || isSubstructure(A.right,B));
    }
    
    private boolean recur(TreeNode A,TreeNode B){
        if(B == null) return true;
        if(A == null || A.val != B.val)  return false;
        return recur(A.left,B.left)&&recur(A.right,B.right);
    }
}
```

### 合并两个排序的链表

```java
//定义一个哑结点
while中是&&
```

### 翻转链表

```java
public ListNode reverseList(ListNode head){
    if(head == null || head.next == null){
        return head;
    }
    ListNode newList = reverseList(head.next);
    head.next.next = head;
    head.next = null;
    return newList;
}
```

### 调整数组顺序使奇数位于偶数前面(双指针)

```java
class Solution{
    public int[] exchange(int[] nums){
        int i = 0, j = nums.length - 1, tmp;
        while(i < j){
        	while(i < j && nums[i] % 2 != 0){
            	i++;
        	}
        	while(i < j && nums[j]%2 == 0){
            	j--;
        	}
        	tmp = nums[i];
            nums[i] = nums[j];
            nums[j] = tmp;
        }
        return nums;
    }
}
```

### 表示数值的字符串（有限状态自动机）-判断字符串是否表示数值

请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串`"+100"`、`"5e2"、"-123"、"3.1416"、"0123"`都表示数值，但`"12e"、"1a3.14"、"1.2.3"、"+-5"、"-1E-16"及"12e+5.4"`都不是。

```java
class Solution{
    public boolean isNumber(String s){
        Map[] states = {
            new HashMap<>() {{put(' ',0);put('s',1);put('d',2);put('.',4);}},      //0
            new HashMap<>() {{put('d',2);put('.',4);}},                            //1
            new HashMap<>() {{put('d',2);put('.',3);put('e',5);put(' ',8);}},       //2
            new HashMap<>() {{put('d',3);put('e',5);put(' ',8);}},                 //3
            new HashMap<>() {{put('d',3);}},
            new HashMap<>() {{ put('s', 6); put('d', 7); }},                        // 5.
            new HashMap<>() {{ put('d', 7); }},                                     // 6.
            new HashMap<>() {{ put('d', 7); put(' ', 8); }},                        // 7.
            new HashMap<>() {{ put(' ', 8); }}                                      // 8.
        };
        int p = 0;
        char t;
        for(char c : s.toCharArray()){
            if(c >= '0' && c <= '9') t = 'd';
            else if(c == '+' || c == '-') t = 's';
            else if(c == '.' || c == 'e' || c == 'E' || c == ' ') t = c;
            else t = '?';
            if(!states[p].containsKey(t)) return false;
            p = (int)states[p].get(t);
        }
        return p == 2 || p == 3 || p == 7 || p == 8;
    }
}
```



### 乘积最大子序列

状态转移方程：$imax=max(imax*nums[i],nums[i])$

$imin=min(imin*nums[i],nums[i])$

```java
class Solution{
    public int maxProduct(int[] nums){
        int max = Integer.MIN_VALUE,imax = 1, imin = 1;
        for(int i=0;i<nums.length;i++){
            if(nums[i] < 0){
                int tmp = imax;
                imax = imin;
                imin = tmp;
            }
            imax = Math.max(imax * nums[i], nums[i]);
            imin = Math.min(imin * nums[i], nums[i]);
            
            max = Math.max(max,imax);
        }
        return max;
    }
}
```

### 两个字符串正则化匹配

```java
class Solution{
    public boolean isMatch(String A,String B){
        int n = A.length();
        int m = B.length();
        boolean[][] f = new boolean[n+1][m+1];
        
        for(int i=0;i<=n;i++){
            for(int j=0;j<=m;j++){
                if(j == 0){
                    f[i][j] = i == 0;
                }else{
                    if(B.charAt(j-1)!='*'){
                        if(i > 0 && (A.charAt(i-1) == B.charAt(j-1) || B.charAt(j-1) == '.')){
                            f[i][j] = f[i-1][j-1];
                        }
                    }else{
                        if(j >= 2){
                            f[i][j] |= f[i][j-2];
                        }
                        if(i >= 1 && j >= 2 && (A.charAt(i-1) == B.charAt(j-2)||B.charAt(j-2) == '.')){
                            f[i][j] |= f[i-1][j];
                        }
                    }
                }
            }
        }
        return f[n][m];
    }
}
```

### 删除链表的结点（利用辅助指针删除）

```##java
class Solution{
    public ListNode deleteNode(ListNode head, int val){
        if(head.val == val) return head.next;
        ListNode pre = head, cur = head.next;
        while(cur != null && cur.val != val){
            pre = cur;
            cur = cur.next;
        }
        pre.next = cur.next;
        return head;
    }
}
```

### 打印从1到最大的n位数（分治算法 / 全排列）

当是很大的数的时候？？？

- 固定高位，进行递归

```java
	static StringBuilder res;
    static int count = 0, n1, nine = 0, start;
    static char[] num, loop = {'0','1','2','3','4','5','6','7','8','9'};
    public static String printNumbers(int n){
        n1 = n;
        res = new StringBuilder();
        num = new char[n1];
        start = n - 1;
        dfs(0);
        res.deleteCharAt(res.length() - 1);
        return res.toString();
    }

    private static void dfs(int x){
        if( x == n1){
            String s = String.valueOf(num).substring(start);
            if (!s.equals("0"))
                res.append(s + ",");
            if (n1 - start == nine) start--;
            return;
        }
        for (char i : loop){
            if(i == '9') nine++;
            num[x] = i;
            dfs(x + 1);
        }
        nine--;
    }
```

### 数值的整数次方（快速幂）

```java
class Solution{
    public double myPow(double x, int n){
        if(x == 0) return 0;
        long b = n;
        double res = 1.0;
        if(b < 0){
            x = 1 / x;
            b = -b;
        }
        while(b > 0){
            if((b & 1) == 1) res *= x;
            x *= x;
            b >>= 1;
        }
        return res;
    }
}
```

**二分法角度**

![image-20200807212506210](C:\Users\xwj\AppData\Roaming\Typora\typora-user-images\image-20200807212506210.png)

答案和上面是一样的

### 剪绳子||（数学推导/贪心思想+快速幂求导）

**推论**：1.当所有绳子长度相等时，乘积最大

2.最优的绳子长度为3

**大数求余发**：

解决方案： 循环求余 、 快速幂求余 ，其中后者的时间复杂度更低，两种方法均基于以下求余运算规则推出：

$(xy) \odot p = [(x \odot p)(y \odot p)] \odot p$

**本题中**：二分求余法

```java
class Solution{
    public int cuttingRope(int n){
        if(n <= 3) return n-1;
        int b = n % 3, p = 1000000007;
        long rem = 1, x = 3;
        for(int a = a / 3-1;)
    }
}
```

**快速幂求余**

```java
class Solution {
    public int cuttingRope(int n) {
        if(n <= 3) return n - 1;
        int b = n % 3, p = 1000000007;
        long rem = 1, x = 3;
        for(int a = n / 3 - 1; a > 0; a /= 2) {
            if(a % 2 == 1) rem = (rem * x) % p;
            x = (x * x) % p;
        }
        if(b == 0) return (int)(rem * 3 % p);
        if(b == 1) return (int)(rem * 4 % p);
        return (int)(rem * 6 % p);
    }
}
```



**循环求余法**（贪心算法）

```java
class Solution{
    if(n <= 3) return n - 1;
    long res = 1L;
    int p = (int)le9+7;
    while(n > 4){
        res = res*3%p;
        n -= 3;
    }
    //出来循环只有三种情况，分别是n=2、3、4
    return (int)(res*n%p);
}
```

### 机器人的运动路径

`地上有一个m行n列的方格，从坐标[0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？`

```java
public class Solution {
    private int[][] direct = {{1, 0}, {0, 1},{-1,0},{0,-1}};
    private boolean[][] visited;
    private int res = 0;
    private int M;
    private int N;

    public int movingCount(int m, int n, int k) {
        this.visited = new boolean[m][n];
        this.M = m;
        this.N = n;
        dfs(0, 0, k);
        return res;
    }

    private void dfs(int x, int y, int k) {
        if (!inArea(x, y) || visited[x][y]) {
            return;
        }
        if (x / 10 + x % 10 + y / 10 + y % 10 <= k) {
            res++;
            visited[x][y] = true;
            for (int i = 0; i < 4; i++) {
                int newX = x + direct[i][0];
                int newY = y + direct[i][1];
                dfs(newX, newY, k);
            }
        }
    }

    private boolean inArea(int x, int y) {
        return x >= 0 && y >= 0 && x < M && y < N;
    }
}
```

### 滑动窗口的最大值

利用一个队列来维护出口最大值，即滑动窗口的最大值

```java
class Solution{
    public int[] maxSlidingWindow(int[] nums, int k){
        if(nums.length == 0 || k == 0) return new int[0];
        Deque<Integer> deque = new LinkedList<>();
        int[] res = new int[nums.length - k + 1];
        for(int j=0,i<1-k;j<nums.length;i++,j++){
            if(i > 0 && deque.peekFirst() == nums[i-1]){
                deque.removeFirst();
            }
            while(!deque.isEmpty() && deque.peekLast() < nums[j]){
                deque.removeLast();
            }
            deque.addLast(nums[j]);
            if(i >= 0){
                res[i] = deque.peekFirst();
            }
            return res;
        }
    }
}
```

### 矩阵中的路径

请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。例如，在下面的3×4的矩阵中包含一条字符串“bfce”的路径（路径中的字母用加粗标出）。

```java
class Solution{
    public boolean exist(char[][] board, String word){
        char[] words = word.toCharArray();
        for(int i=0;i<board.length;i++){
            for(int j=0;j<board[0].length;j++){
                if(dfs(board,words,i,j,0)) return true;
            }
        }
        return false;
    }
    boolean dfs(char[][] board, char[] word,int i, int j,int k){
        if(i >= board.length || i < 0 || j >= board[0].length || j < 0 || board[i][j] != word[k]) return false;
        if(k == word.length - 1) return true;
        char tmp = board[i][j];
        board[i][j] = '/';
        boolean res = dfs(board,word,i+1,j,k+1) ||dfs(board,word,i-1,j,k+1) ||dfs(board,word,i,j+1,k+1) ||dfs(board,word,i,j-1,k+1);
        board[i][j] = tmp;
        return res;
    }
}
```

### 旋转数组的最小数字

```java
class Solution{
    public int minArray(int[] numbers){
        int i=0, j = numbers.length - 1;
        while(i < j){
            int m = (i + j) / 2;
            if(numbers[m] > numbers[j]) i = m+1;
            else if(numbers[m] < numbers[j]) j = m;
            else j--;
        }
    }
}
```

### 数组中重复的数字

**思路**：原地置换

如果没有重复数字，那么正常排序后，数字i应该在下标i的位置，所以思路就是重头扫描数组，遇到下标为i的数字如果不是i的话，`nums[i]`与`nums[nums[i]]`进行交换。

```java
class Solution{
    public int findRepeatNumber(int[] nums){
        int temp;
        for(int i=0;i< nums.length;i++){
            while(nums[i]!=i){
                if(nums[i] == nums[nums[i]]){
                    return nums[i];
                }
                temp = nums[i];
                nums[i] = nums[temp];
                nums[temp] = temp;
            }
        }
        return -1;
    }
}
```

### 缺失的第一个整数

```java
class Solution{
    public int firstMissingPositive(int[] nums){
        int len = nums.length;
        for(int i=0;i<len;i++){
            while(nums[i]>0&nums[i] <=len&&nums[nums[i]-1]!=nums[i]){
                swap(nums,nums[i]-1,i);
            }
        }
        for(int i=0;i<len;i++){
            if(nums[i] != i+1){
                return i+1;
            }
        }
        return len + 1;
    }
    private void swap(int[] nums,int index1,int index2){
        int temp = nums[index1];
        nums[index1] = nums[index2];
        nums[index2] = temp;
    }
}
```

### 跳台阶

一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

```java
class Solution{
    public int numWays(int n){
        if(n == 0) return 1;
        if(n == 1) return 1;
        int[] dp = new int[n+1];
        dp[0] = 1;
        dp[1] = 1;
        for(int i=2;i<=n;i++){
            dp[i] = (dp[i-1] +dp[i-2]) % 1000000007;
        }
        return dp[n];
    }
}
```



### 斐波那契数列

**思路**：初始化三个整形遍历`sum`，`a`，`b`，利用辅助变量`sum`使得`a`，`b`两个数字交替前进即可，这样可以节约`O(n)`的空间。

循环求余法，规则：sum = (a + b)$\bigodot$p

```java
class Solution{
    public int fib(int n){
        int a = 0, b = 1;
        for(int i=0;i<n;i++){
            int sum = (a + b) % 1000000007;
            a = b;
            b = sum;
        }
        return a;
    }
}
```

### 使用两个栈实现队列

```java
class CQueue{
    LinkedList<Integer> A,B;
    public CQueue(){
        A = new LinkedList<>();
        B = new LinkedList<>();
    }
    public void appendTail(int value){
        A.push(value);
    }
    public int deleteHead(){
        if(!B.isEmpty()){
            return B.pop();
        }
        if(A.isEmpty()) return -1;
        while(!A.isEmpty()){
            B.push(A.pop());
        }
        return B.pop();
    }
}
```

### 重建二叉树

输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。

```java
class Solution{
    public TreeNode buildTree(int[] preorder,int[] inorder){
        if(preorder.length == 0||inorder.length == 0) return null;
        return dfs(preorder,inorder,0,0,inorder.length-1);
    }
    public TreeNode dfs(int[] preorder,int[] inorder,int preStart,int inStart,int inEnd){
        if(inStart > inEnd) return null;
        int currentVal = preorder[preStart];
        TreeNode root = new TreeNode(currentVal);
        int index = 0;
        for(int i=0;i<inorder.length;i++){
            if(currentVal == inorder[i]){
                index = i;
            }
        }
        root.left = dfs(preorder,inorder,preStart+1,inStart,index-1);
        root.right = dfs(preorder,inorder,preStart+index-inStart+1,index+1,inEnd);
        return root;
    }
}
```

**二 ：**上面那种方法需要遍历整个中序序列，所有可以用map来规划

```java
class Solution{
    HashMap<Integer,Integer> dict = new HashMap<>();
    int[] po;
    public TreeNode buildTree(int[] preorder,int[] inorder){
        po = preorder;
        for(int i=0;i<inorder.length;i++){
            dic.put(inorder[i],i);
        }
        return recur(0,0,inorder.length-1);
    }
    public TreeNode recur(int left_root, int in_left,int in_right){
        if(in_left > in_right) return null;
        TreeNode root = new TreeNode(po[left_root]);
        int i = map.get(po[left_root]);
        root.left = recur(left_root + 1,in_left,i-1);
        root.right =recur(left_root+i-in_left+1,i+1,in_right);
        return root;
    }
}
```

该方法利用一个全局的数组存储前序遍历，利用`HashMap`对中序遍历的值和下标建立`key-value`映射。所以在递归遍历的时候不需要再传入前序和中序的数组，直接是下标即可。

前序+中序是：left_root + (i-in_left) + 1

后序+中序是：right_root - (in_right-i)-1

### 重头到尾打印链表

输入一个链表的头结点，从尾到头反过来返回每个节点的值（用数组返回）。

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    List<Integer> list = new ArrayList<>();
    public int[] reversePrint(ListNode head) {
        dfs(head);
        int[] arr = new int[list.size()];
        for(int i=0;i<list.size();i++){
            arr[i] = list.get(i);
        }
        return arr;
    }
    public void dfs(ListNode head){
        if(head == null) return;
        dfs(head.next);
        list.add(head.val);
    }
}
```

### 替换空格

请实现一个函数，把字符串 `s` 中的每个空格替换成"%20"。

```java
class Solution {
    public String replaceSpace(String s) {
        StringBuilder res = new StringBuilder();
        for(char c:s.toCharArray()){
            if(c == ' '){
                res.append("%20");
            }
            else{
                res.append(c);
            }
        }
        return res.toString();
    }
}
```

### 二维数组中的查找

在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

```java
class Solution {
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        int i = matrix.length - 1, j = 0;
        while(i >= 0 && j < matrix[0].length)
        {
            if(matrix[i][j] > target) i--;
            else if(matrix[i][j] < target) j++;
            else return true;
        }
        return false;
    }
}
```

### 数组中重复的数字

**方法一**：利用Set

```java
class Solution{
    public int findRepeatNumber(int[] nums){
        HashSet<Integer> set = new HashSet<>();
        for(int num : nums){
            if(set.contains(num)) return num;
            set.add(num);
        }
        return -1;
    }
}
```

**方法二**：原地交换

```java
class Solution{
    public int findRepeatNumber(int[] nums){
        int i = 0;
        while(i < nums.length){
            if(nums[i] == i){
                i++;
                continue;
            }
            if(nums[nums[i]] == nums[i]) return nums[i];
            int tmp = nums[i];
            nums[i] = nums[tmp];
            nums[tmp] = tmp;
        }
        return -1;
    }
}
```













